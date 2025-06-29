import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger # type: ignore
from model import build_unet_model
import threading
import psutil # type: ignore
import time

# Constants
IMAGE_H = 256
IMAGE_W = 256
MAX_CPU_PERCENT = 80
MAX_RAM_GB = 4
MAX_GPU_PERCENT = 80

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)]
                )
            print("GPU configured")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

def setup_cpu():
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(4)

def monitor_resources():
    def resource_monitor():
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                ram_gb = memory.used / (1024**4)
                
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = gpu_util.gpu
                    
                    if gpu_percent > MAX_GPU_PERCENT:
                        print(f"Warning: GPU usage high: {gpu_percent}%")
                except:
                    gpu_percent = 0
                
                if cpu_percent > MAX_CPU_PERCENT:
                    print(f"Warning: CPU usage high: {cpu_percent}%")
                    
                if ram_gb > MAX_RAM_GB:
                    print(f"Warning: RAM usage high: {ram_gb:.1f}GB")
                
                if hasattr(monitor_resources, 'verbose') and monitor_resources.verbose:
                    print(f"Stats - CPU: {cpu_percent}% | RAM: {ram_gb:.1f}GB | GPU: {gpu_percent}%")
                
                time.sleep(30)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(60)
    
    monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def load_dataset(path, split=0.1):
    images_path = os.path.join(path, "images", "*.jpg")
    masks_path = os.path.join(path, "masks", "*.png")
    
    X = sorted(glob(images_path))
    Y = sorted(glob(masks_path))
    
    if not X or not Y:
        raise ValueError(f"No images found in {path}")
    
    if len(X) != len(Y):
        raise ValueError(f"Mismatch: {len(X)} images vs {len(Y)} masks")
    
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=split, random_state=42)
    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    if isinstance(path, bytes):
        path = path.decode()
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    
    img = cv2.resize(img, (IMAGE_W, IMAGE_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img / 255.0).astype(np.float32)

def read_mask(path):
    if isinstance(path, bytes):
        path = path.decode()
    
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    
    mask = cv2.resize(mask, (IMAGE_W, IMAGE_H))
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask, mask], axis=-1)
    return mask

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMAGE_H, IMAGE_W, 3])
    y.set_shape([IMAGE_H, IMAGE_W, 4])
    return x, y

def create_dataset(X, Y, batch_size=2, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 1000))
    
    ds = ds.map(tf_parse, num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(2)
    
    return ds

def main():
    print("Configuring hardware...")
    setup_gpu()
    setup_cpu()
    
    print("Starting resource monitor...")
    monitor_thread = monitor_resources()
    monitor_resources.verbose = True
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    create_dir("files")
    
    input_shape = (IMAGE_H, IMAGE_W, 3)
    batch_size = 2
    lr = 1e-4
    num_epochs = 100
    
    dataset_path = r'people_segmentation_edited'
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "training_log.csv")
    
    try:
        print("Loading dataset...")
        (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path, split=0.2)
        print(f"Train: {len(train_x)} samples - Valid: {len(valid_x)} samples")
        
        print("Creating data pipelines...")
        train_ds = create_dataset(train_x, train_y, batch_size, shuffle=True)
        valid_ds = create_dataset(valid_x, valid_y, batch_size, shuffle=False)
        
        print("Building model...")
        model = build_unet_model(input_shape)
        
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled")
        except:
            print("Mixed precision not available")
        
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy"]
        )
        
        model.summary()
        
        callbacks = [
            ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
            CSVLogger(csv_path),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        ]
        
        print("Starting training...")
        print(f"Target usage: CPU<{MAX_CPU_PERCENT}%, RAM<{MAX_RAM_GB}GB, GPU<{MAX_GPU_PERCENT}%")
        
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=num_epochs,
            callbacks=callbacks,
            verbose=1,
            workers=1,
            use_multiprocessing=False
        )
        
        print("Training completed!")
        print(f"Model saved: {model_path}")
        print(f"Log saved: {csv_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    finally:
        monitor_resources.verbose = False
        tf.keras.backend.clear_session()
        print("Cleanup completed")

if __name__ == "__main__":
    main()