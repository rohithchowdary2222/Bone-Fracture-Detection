import os
import numpy as np
import cv2
import pickle

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, VGG19
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    HAS_TF = True
except ImportError:
    print("Warning: TensorFlow not found. Using mock feature extraction for demonstration.")
    HAS_TF = False

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "dataset/" # Assumed structure: dataset/train/fractured, dataset/train/normal

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    # Noise reduction (Median Blur)
    img = cv2.medianBlur(img, 3)
    img = img / 255.0  # Normalization
    return img

def create_feature_extractor(model_name='EfficientNetB0'):
    if not HAS_TF:
        return None
    if model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(*IMG_SIZE, 3))
    elif model_name == 'MobileNet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(*IMG_SIZE, 3))
    elif model_name == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False, pooling='avg', input_shape=(*IMG_SIZE, 3))
    return base_model

def extract_features(model, data_flow):
    if not HAS_TF:
        # Return random features for demonstration if TF is missing
        num_samples = len(data_flow.filenames)
        return np.random.rand(num_samples, 1280) # EfficientNetB0 output size
    features = model.predict(data_flow)
    return features

# --- Mock Data Generator if TF missing ---
class MockDataFlow:
    def __init__(self, directory):
        self.filenames = []
        self.classes = []
        self.class_indices = {'fractured': 1, 'normal': 0}
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.filenames.append(os.path.join(root, file))
                    self.classes.append(1 if 'fractured' in root else 0)

# --- Hybrid Pipeline ---
def train_hybrid_model():
    # 1. Image Data Generator
    if HAS_TF:
        datagen = ImageDataGenerator(rescale=1./255)
        train_generator = datagen.flow_from_directory(
            'dataset/train',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
    else:
        train_generator = MockDataFlow('dataset/train')

    print("Loading feature extractor (EfficientNetB0)...")
    feature_extractor = create_feature_extractor('EfficientNetB0')

    print("Extracting features from training data...")
    X_train_features = extract_features(feature_extractor, train_generator)
    y_train = train_generator.classes

    # --- XGBoost Model ---
    print("Training XGBoost classifier...")
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_features, y_train)
    
    # Save model and feature extractor info
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/efficientnet_xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Save classes for reference
    with open('models/classes.pkl', 'wb') as f:
        pickle.dump(train_generator.class_indices, f)
        
    print("Hybrid model (EfficientNet + XGBoost) trained and saved to models/efficientnet_xgboost.pkl.")

if __name__ == "__main__":
    train_hybrid_model()
