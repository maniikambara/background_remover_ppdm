# type: ignore
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import gc
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from model import build_model

# Optimized settings for mid-range laptop
IMAGE_H = 256  # Reduced from 512
IMAGE_W = 256  # Reduced from 512

def setup_gpu():
    """Configure GPU for optimal memory usage"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Limit GPU memory to 4GB
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], 
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
            print(f"GPU configured: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU detected, using CPU")

def create_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def load_dataset(path, split=0.2, max_samples=1000):
    """Load dataset with memory optimization"""
    images = sorted(glob(os.path.join(path, "images", "*.jpg")))
    masks = sorted(glob(os.path.join(path, "masks", "*.png")))
    
    if len(images) == 0 or len(masks) == 0:
        raise ValueError(f"No images or masks found in {path}")
    
    # Limit dataset size for memory efficiency
    if len(images) > max_samples:
        images = images[:max_samples]
        masks = masks[:max_samples]
        print(f"Dataset limited to {max_samples} samples for memory efficiency")
    
    if len(images) != len(masks):
        min_len = min(len(images), len(masks))
        images = images[:min_len]
        masks = masks[:min_len]
        print(f"Matched dataset to {min_len} samples")
    
    # Split data
    train_x, valid_x, train_y, valid_y = train_test_split(
        images, masks, test_size=split, random_state=42, shuffle=True
    )
    
    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    """Read and preprocess image with memory optimization"""
    if isinstance(path, bytes):
        path = path.decode()
    
    # Read with lower quality for memory efficiency
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    
    # Resize to smaller dimensions
    img = cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    
    return img

def read_mask(path):
    """Read and preprocess mask with memory optimization"""
    if isinstance(path, bytes):
        path = path.decode()
    
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    
    # Resize to smaller dimensions
    mask = cv2.resize(mask, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)
    
    # Convert to 4 channels to match model output
    mask = np.concatenate([mask, mask, mask, mask], axis=-1)
    return mask

def tf_parse(x, y):
    """Optimized parse function"""
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMAGE_H, IMAGE_W, 3])
    y.set_shape([IMAGE_H, IMAGE_W, 4])
    return x, y

def create_dataset(X, Y, batch_size=2, shuffle=True, cache=False):
    """Create memory-efficient dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)  # Reduced buffer
    
    dataset = dataset.map(tf_parse, num_parallel_calls=2)  # Limited parallel calls
    dataset = dataset.batch(batch_size)
    
    if cache and len(X) < 500:  # Only cache small datasets
        dataset = dataset.cache()
    
    dataset = dataset.prefetch(2)  # Reduced prefetch buffer
    
    return dataset

def clear_memory():
    """Clear memory to prevent OOM"""
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()

def main():
    # Setup GPU first
    setup_gpu()
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory
    create_dir("files")
    
    # Optimized hyperparameters for mid-range laptop
    input_shape = (IMAGE_H, IMAGE_W, 3)
    batch_size = 2  # Small batch size for RTX 4050
    learning_rate = 1e-4
    epochs = 50  # Reduced epochs
    max_samples = 800  # Limit dataset size
    
    # Paths
    dataset_path = "./people_segmentation_edited"
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "training_log.csv")
    
    try:
        # Load dataset with limits
        print("Loading dataset...")
        (train_x, train_y), (valid_x, valid_y) = load_dataset(
            dataset_path, split=0.2, max_samples=max_samples
        )
        print(f"Train: {len(train_x)} samples - Valid: {len(valid_x)} samples")
        print(f"Image size: {IMAGE_H}x{IMAGE_W}")
        print(f"Batch size: {batch_size}")
        
        # Create datasets
        print("Creating data pipelines...")
        train_ds = create_dataset(train_x, train_y, batch_size, shuffle=True)
        valid_ds = create_dataset(valid_x, valid_y, batch_size, shuffle=False)
        
        # Build model with mixed precision for efficiency
        print("Building model...")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        model = build_model(input_shape)
        
        # Use mixed precision optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        
        print(f"Model parameters: {model.count_params():,}")
        
        # Memory-efficient callbacks
        callbacks = [
            ModelCheckpoint(
                model_path, 
                monitor='val_loss', 
                save_best_only=True, 
                verbose=1,
                save_weights_only=False
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=3,  # Reduced patience
                min_lr=1e-7, 
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss', 
                patience=8,  # Reduced patience
                restore_best_weights=True,
                verbose=1
            ),
            CSVLogger(csv_path)
        ]
        
        # Training with memory monitoring
        print("Starting training...")
        print("Monitoring memory usage...")
        
        # Clear memory before training
        clear_memory()
        
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            workers=2,  # Limited workers
            use_multiprocessing=False  # Disable multiprocessing to save RAM
        )
        
        print("Training completed!")
        print(f"Model saved to: {model_path}")
        print(f"Training log saved to: {csv_path}")
        
        # Final memory cleanup
        clear_memory()
        
    except Exception as e:
        print(f"Error: {e}")
        clear_memory()
        return False
    
    return True

if __name__ == "__main__":
    success = main()