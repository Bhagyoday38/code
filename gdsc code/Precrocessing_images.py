import os
import pickle
import warnings
import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
warnings.filterwarnings("ignore", category=UserWarning, module='absl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r'C:\Users\Avinash\Desktop\sign-language-detector-python\sign-language-detector-python-2\data'
IMG_SIZE = (128, 128)  # Resize images to this size

def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    
    results = hands.process(img_resized)
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
        
        if len(landmarks) < 84:
            landmarks += [0] * (84 - len(landmarks))  # Pad with zeros if less than 84 features
        
        landmarks_np = np.array(landmarks[:84])
        landmarks_normalized = (landmarks_np - landmarks_np.min()) / (landmarks_np.max() - landmarks_np.min())
        
        return landmarks_normalized
    else:
        return np.zeros(84)  # Return a zero vector if no hand is detected

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    subdir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(subdir_path):
        continue

    for img_path in os.listdir(subdir_path):
        img_full_path = os.path.join(subdir_path, img_path)
        img = cv2.imread(img_full_path)
        
        if img is None:
            print(f"Warning: Failed to load image at {img_full_path}")
            continue

        features = preprocess_image(img)
        data.append(features)
        labels.append(dir_)

data_np = np.array(data)
labels_np = np.array(labels)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels_np)
encoded_labels_categorical = to_categorical(encoded_labels)

with open('preprocessed_data.pickle', 'wb') as f:
    pickle.dump({'data': data_np, 'labels': encoded_labels_categorical}, f)

print("Data preprocessing completed.")
