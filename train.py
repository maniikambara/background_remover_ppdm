# WARNING: JANGAN RUN CODE INI DI LAPTOP DENGAN SPEC MENENGAH! (CPU 4GHZ, RAM 16GB, VRAM6GB) RUN CODE INI JIKA LAPTOP ANDA RAM DIATAS 32GB DAN VRAM DIATAS 8GB!

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger # type: ignore
from model import build_unet_model

# Global constants
IMAGE_H = 512
IMAGE_W = 512

def create_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def load_dataset(path, split=0.1):
    """Load images and masks with proper validation"""
    images_path = os.path.join(path, "images", "*.jpg")
    masks_path = os.path.join(path, "masks", "*.png")
    
    X = sorted(glob(images_path))
    Y = sorted(glob(masks_path))
    
    if not X or not Y:
        raise ValueError(f"No images found in {path}. Check directory structure.")
    
    if len(X) != len(Y):
        raise ValueError(f"Mismatch: {len(X)} images vs {len(Y)} masks")
    
    # Split data
    train_x, valid_x, train_y, valid_y = train_test_split(
        X, Y, test_size=split, random_state=42
    )
    
    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    """Read and preprocess image"""
    if isinstance(path, bytes):
        path = path.decode()
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    
    img = cv2.resize(img, (IMAGE_W, IMAGE_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return (img / 255.0).astype(np.float32)

def read_mask(path):
    """Read and preprocess mask"""
    if isinstance(path, bytes):
        path = path.decode()
    
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    
    mask = cv2.resize(mask, (IMAGE_W, IMAGE_H))
    mask = mask.astype(np.float32)
    
    # Expand dims and concatenate to 4 channels like original
    mask = np.expand_dims(mask, axis=-1)  # (h, w, 1)
    mask = np.concatenate([mask, mask, mask, mask], axis=-1)  # (h, w, 4)
    return mask

def tf_parse(x, y):
    """TensorFlow parsing function"""
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMAGE_H, IMAGE_W, 3])
    y.set_shape([IMAGE_H, IMAGE_W, 4])
    return x, y

def create_dataset(X, Y, batch_size=2, shuffle=True):
    """Create optimized TensorFlow dataset"""
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    
    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def main():
    """Main training function"""
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory
    create_dir("files")
    
    # Hyperparameters
    input_shape = (IMAGE_H, IMAGE_W, 3)
    batch_size = 4
    lr = 1e-4
    num_epochs = 100
    
    # Paths
    dataset_path = r'people_segmentation_edited'
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "training_log.csv")
    
    try:
        # Load dataset
        print("Loading dataset...")
        (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path, split=0.2)
        print(f"Train: {len(train_x)} samples - Valid: {len(valid_x)} samples")
        
        # Create datasets
        print("Creating data pipelines...")
        train_ds = create_dataset(train_x, train_y, batch_size, shuffle=True)
        valid_ds = create_dataset(valid_x, valid_y, batch_size, shuffle=False)
        
        # Build model
        print("Building model...")
        model = build_unet_model(input_shape)
        
        # Compile model
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy"]
        )
        
        # Print model summary
        model.summary()
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                model_path, 
                monitor='val_loss', 
                verbose=1, 
                save_best_only=True,
                save_weights_only=False
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-7, 
                verbose=1
            ),
            CSVLogger(csv_path),
            EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        print("Starting training...")
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=num_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Training log saved to: {csv_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()