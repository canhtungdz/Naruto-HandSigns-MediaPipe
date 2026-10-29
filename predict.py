import os
import cv2
import json
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import extract_hand_features
from train_mlp import NarutoMLP

def run_predictions():
    # 1. Load mappings
    with open("data/class_mapping.json", "r") as f:
        class_mapping = json.load(f)
        
    num_classes = len(class_mapping)
    
    # 2. Setup model and device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NarutoMLP(input_size=88, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("data/mlp_model.pth", map_location=device))
    model.eval()
    
    # 3. Setup MediaPipe Hands Tasks API
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    temp_dir = "./temp"
    images = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print("No images found in the temp directory!")
        return
        
    print(f"Detecting hands and predicting signs on {len(images)} images in {temp_dir} using device: {device}...\n")
    
    for img_name in sorted(images):
        img_path = os.path.join(temp_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{img_name}] Error: Could not read image.")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Predict using Tasks API
        results = detector.detect(mp_image)
        
        if not results.hand_landmarks:
            print(f"[{img_name}] Result: No hand detected.")
            continue
            
        # Get landmarks
        hand_lms = results.hand_landmarks[0]
        raw_coords = []
        for lm in hand_lms:
            raw_coords.extend([lm.x, lm.y, lm.z])
            
        # Extract features (angles, distances, relative coordinates)
        features = extract_hand_features(raw_coords, include_coords=True)
        
        # Run model prediction
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prob, predicted = torch.max(probabilities, 1)
            
        class_id = str(predicted.item())
        class_name = class_mapping[class_id]
        
        print(f"[{img_name}] Prediction: {class_name} | Confidence: {prob.item() * 100:.2f}%")

if __name__ == "__main__":
    run_predictions()
