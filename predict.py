import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf

# Global parameters
image_h = 256
image_w = 256

def create_dir(path):
    """Create directory if not exists"""
    os.makedirs(path, exist_ok=True)

def preprocess_image(image_path):
    """Load and preprocess image"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    h, w = image.shape[:2]
    x = cv2.resize(image, (image_w, image_h))
    x = (x / 255.0).astype(np.float32)
    return image, np.expand_dims(x, 0), (w, h)

def postprocess_mask(prediction, original_size):
    """Process model prediction to mask"""
    mask = prediction[0][:, :, -1] if prediction.ndim == 4 else prediction[0]
    mask = cv2.resize(mask, original_size)
    
    # Convert to 0-255 range and ensure uint8 type
    mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
    return np.expand_dims(mask, -1)

def create_result_image(original, mask):
    """Create side-by-side result image"""
    # Ensure mask is in 0-1 range for multiplication
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # Apply mask to original image
    masked = (original * mask_normalized).astype(np.uint8)
    
    h = original.shape[0]
    divider = np.full((h, 10, 3), 128, dtype=np.uint8)
    
    return np.concatenate([original, divider, masked], axis=1)

def main():
    # Setup
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory
    create_dir(r'test/masks')
    
    # Load model
    try:
        model = tf.keras.models.load_model(r'files/model.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get test images
    image_paths = glob(r'test/images/*')
    if not image_paths:
        print("No images found in the folder")
        return
    
    print(f"Processing {len(image_paths)} images...")
    
    # Process each image
    for path in tqdm(image_paths):
        try:
            # Extract filename
            name = os.path.splitext(os.path.basename(path))[0]
            
            # Process image
            original, processed, size = preprocess_image(path)
            
            # Predict
            prediction = model.predict(processed, verbose=0)
            
            # Create mask
            mask = postprocess_mask(prediction, size)
            
            # Create and save result
            result = create_result_image(original, mask)
            output_path = f"test/masks/{name}.png"
            
            # Ensure result is uint8 before saving
            if result.dtype != np.uint8:
                result = result.astype(np.uint8)
            
            cv2.imwrite(output_path, result)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    print("Processing completed!")

if __name__ == "__main__":
    main()