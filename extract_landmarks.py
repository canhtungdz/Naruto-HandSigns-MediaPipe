import os
import json
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

def get_class_mapping(data_dir):
    """
    Scans the YOLO label files to dynamically map class_id (int) to class_name (str)
    based on the filename prefixes (e.g. bird_xxx.txt has class_id 0).
    """
    print("Scanning dataset to map class IDs to class names...")
    mapping = {}
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        labels_path = os.path.join(data_dir, split, 'labels')
        if not os.path.isdir(labels_path):
            continue
            
        for f in os.listdir(labels_path):
            if f.endswith('.txt'):
                class_name = f.split('_')[0]
                full_path = os.path.join(labels_path, f)
                try:
                    with open(full_path, 'r') as file:
                        for line in file:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                mapping[class_id] = class_name
                except Exception:
                    pass
    
    # Sort mapping by class_id
    sorted_mapping = {k: mapping[k] for k in sorted(mapping.keys())}
    print(f"Detected class mapping: {sorted_mapping}")
    return sorted_mapping

def process_dataset(data_dir, output_dir, save_crops=True):
    """
    Main pipeline to process train, valid, and test sets.
    Extracts landmarks, normalizes them, and saves the data as CSVs.
    """
    # 1. Get class mapping
    class_mapping = get_class_mapping(data_dir)
    if not class_mapping:
        print("Error: No class labels found. Please check your data directory.")
        return
        
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Save class mapping to JSON for future inference use
    with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=4)
        
    splits = ['train', 'valid', 'test']
    
    # Initialize MediaPipe Hand landmarker
    # We set static_image_mode=True as we are processing individual images
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    ) as hands:
        
        for split in splits:
            images_dir = os.path.join(data_dir, split, 'images')
            labels_dir = os.path.join(data_dir, split, 'labels')
            
            if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
                print(f"Skipping split '{split}' (directories not found).")
                continue
                
            print(f"\nProcessing split: {split}")
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            records = []
            failed_detections = {} # Track failure rates per class
            total_per_class = {}
            
            for img_name in tqdm(image_files, desc=f"Extracting {split}"):
                img_path = os.path.join(images_dir, img_name)
                
                # Retrieve corresponding label file
                base_name = os.path.splitext(img_name)[0]
                label_path = os.path.join(labels_dir, f"{base_name}.txt")
                
                if not os.path.exists(label_path):
                    continue
                    
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h_img, w_img, _ = img.shape
                
                # Parse YOLO label
                with open(label_path, 'r') as f_label:
                    lines = f_label.readlines()
                    
                if not lines:
                    continue
                    
                # We process the first detected bounding box in the file (main hand sign)
                parts = lines[0].strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                class_name = class_mapping.get(class_id, "unknown")
                
                # Keep statistics
                total_per_class[class_name] = total_per_class.get(class_name, 0) + 1
                
                x_center = float(parts[1]) * w_img
                y_center = float(parts[2]) * h_img
                width = float(parts[3]) * w_img
                height = float(parts[4]) * h_img
                
                # Bounding box coordinates
                x_min = int(x_center - width/2)
                y_min = int(y_center - height/2)
                x_max = int(x_center + width/2)
                y_max = int(y_center + height/2)
                
                # Add 20% padding to bounding box to help MediaPipe detect hand contextual features
                pad_w = int(width * 0.20)
                pad_h = int(height * 0.20)
                x_min_pad = max(0, x_min - pad_w)
                y_min_pad = max(0, y_min - pad_h)
                x_max_pad = min(w_img, x_max + pad_w)
                y_max_pad = min(h_img, y_max + pad_h)
                
                # Crop hand area
                crop = img[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
                if crop.size == 0:
                    crop = img # fallback to full image if crop is invalid
                    h_crop, w_crop = h_img, w_img
                else:
                    h_crop, w_crop = crop.shape[0], crop.shape[1]
                    
                # Convert BGR to RGB for MediaPipe
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                # Run hand landmark detection on the crop
                results = hands.process(crop_rgb)
                
                # Fallback to full image if detection on crop failed
                if not results.multi_hand_landmarks:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)
                    w_crop, h_crop = w_img, h_img # update dimensions to full image
                    
                # If hand is detected
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    
                    # 1. Raw Landmark Coordinates (relative to the crop/image size)
                    raw_coords = []
                    for lm in hand_lms.landmark:
                        raw_coords.extend([lm.x, lm.y, lm.z])
                        
                    # 2. Convert to pixel scale on the crop to calculate distance metric accurately
                    pixel_coords = []
                    for lm in hand_lms.landmark:
                        pixel_coords.append([lm.x * w_crop, lm.y * h_crop, lm.z * w_crop])
                    pixel_coords = np.array(pixel_coords)
                    
                    # 3. Calculate Relative Coordinates (subtract wrist coordinates - landmark 0)
                    wrist = pixel_coords[0]
                    rel_coords = pixel_coords - wrist
                    
                    # 4. Calculate Scale-Normalized Coordinates (divide relative coordinates by hand scale)
                    # Hand scale is calculated as the Euclidean distance between wrist (0) and middle finger MCP joint (9)
                    hand_scale = np.linalg.norm(rel_coords[9])
                    if hand_scale == 0:
                        hand_scale = 1.0 # Prevent division by zero
                    norm_coords = rel_coords / hand_scale
                    
                    # Store record
                    row = {
                        'filename': img_name,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    
                    # Add features to the row
                    for idx in range(21):
                        # Raw coordinates (0 to 1)
                        row[f'raw_x_{idx}'] = raw_coords[idx*3]
                        row[f'raw_y_{idx}'] = raw_coords[idx*3 + 1]
                        row[f'raw_z_{idx}'] = raw_coords[idx*3 + 2]
                        
                        # Relative coordinates
                        row[f'rel_x_{idx}'] = rel_coords[idx][0]
                        row[f'rel_y_{idx}'] = rel_coords[idx][1]
                        row[f'rel_z_{idx}'] = rel_coords[idx][2]
                        
                        # Normalized coordinates (translation and scale invariant)
                        row[f'norm_x_{idx}'] = norm_coords[idx][0]
                        row[f'norm_y_{idx}'] = norm_coords[idx][1]
                        row[f'norm_z_{idx}'] = norm_coords[idx][2]
                        
                    records.append(row)
                    
                    # Optional: Save cropped hand image for verification
                    if save_crops:
                        crop_out_dir = os.path.join(output_dir, 'images', split, class_name)
                        os.makedirs(crop_out_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(crop_out_dir, img_name), crop)
                else:
                    # Keep track of failed detections
                    failed_detections[class_name] = failed_detections.get(class_name, 0) + 1
                    
            # Save extracted landmarks to CSV
            if records:
                df = pd.DataFrame(records)
                csv_path = os.path.join(output_dir, f"{split}.csv")
                df.to_csv(csv_path, index=False)
                print(f"Saved {len(records)} samples to {csv_path}")
            else:
                print(f"Warning: No hands detected in split '{split}'!")
                
            # Print detection statistics for this split
            print(f"--- Detection Rate Report for '{split}' ---")
            for c_name, total in total_per_class.items():
                failed = failed_detections.get(c_name, 0)
                detected = total - failed
                rate = (detected / total) * 100 if total > 0 else 0
                print(f" Class '{c_name}': Detected {detected}/{total} ({rate:.2f}%)")

if __name__ == "__main__":
    DATA_DIR = "./data"
    OUTPUT_DIR = "./data"
    process_dataset(DATA_DIR, OUTPUT_DIR, save_crops=True)
