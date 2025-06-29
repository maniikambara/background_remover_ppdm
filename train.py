# type: ignore
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from model import build_model

# Global constants
IMAGE_H = 512
IMAGE_W = 512

def create_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def load_dataset(path, split=0.2):
    """Load images and masks with proper validation"""
    images = sorted(glob(os.path.join(path, "images", "*.jpg")))
    masks = sorted(glob(os.path.join(path, "masks", "*.png")))
    
    if len(images) == 0 or len(masks) == 0:
        raise ValueError(f"No images or masks found in {path}")
    
    if len(images) != len(masks):
        raise ValueError(f"Mismatch: {len(images)} images, {len(masks)} masks")
    
    # Ensure minimum split size
    min_valid_size = max(1, int(len(images) * split))
    if min_valid_size >= len(images):
        raise ValueError(f"Dataset too small: {len(images)} samples")
    
    # Split data
    train_x, valid_x, train_y, valid_y = train_test_split(
        images, masks, test_size=split, random_state=42, shuffle=True
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
    img = img.astype(np.float32) / 255.0
    return img

def read_mask(path):
    """Read and preprocess mask"""
    if isinstance(path, bytes):
        path = path.decode()
    
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    
    mask = cv2.resize(mask, (IMAGE_W, IMAGE_H))
    mask = mask.astype(np.float32) / 255.0  # Normalize mask
    mask = np.expand_dims(mask, axis=-1)  # (h, w, 1)
    
    # Convert to 4 channels to match model output
    mask = np.concatenate([mask, mask, mask, mask], axis=-1)  # (h, w, 4)
    return mask

def tf_parse(x, y):
    """Parse function for TensorFlow dataset"""
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMAGE_H, IMAGE_W, 3])
    y.set_shape([IMAGE_H, IMAGE_W, 4])  # Fixed: mask should match model output (4 channels)
    return x, y

def create_dataset(X, Y, batch_size=4, shuffle=True):
    """Create optimized TensorFlow dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory
    create_dir("files")
    
    # Hyperparameters
    input_shape = (IMAGE_H, IMAGE_W, 3)
    batch_size = 4
    learning_rate = 1e-4
    epochs = 100
    
    # Paths
    dataset_path = "./people_segmentation_edited"
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
        model = build_model(input_shape)
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=["accuracy"]
        )
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                model_path, 
                monitor='val_loss', 
                save_best_only=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-7, 
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True,
                verbose=1
            ),
            CSVLogger(csv_path)
        ]
        
        # Training
        print("Starting training...")
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        print(f"Model saved to: {model_path}")
        print(f"Training log saved to: {csv_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()