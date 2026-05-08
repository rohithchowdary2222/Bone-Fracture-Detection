import cv2
import numpy as np

IMG_SIZE = (224, 224)

def preprocess_image(img_path):
    """
    Standard preprocessing for X-ray images.
    1. Read and convert to RGB
    2. Resize to 224x224
    3. Noise reduction (Median Blur)
    4. Normalize pixel values
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError("Could not read image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    # Noise reduction
    img = cv2.medianBlur(img, 3)
    img = img / 255.0  # Normalization
    return img

def image_to_array(img):
    """Convert preprocessed image to a 4D array for CNN input."""
    return np.expand_dims(img, axis=0)
