import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import feature extractor helper
from features import get_angle, get_tip_distances

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class HandSignsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NarutoMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NarutoMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

def load_and_extract_features(csv_path, include_coords=True):
    """
    Loads landmark CSV, extracts joint angles and distances, and returns feature array and targets.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Select columns with scale-normalized coordinates (63 values)
    norm_cols = []
    for i in range(21):
        norm_cols.extend([f'norm_x_{i}', f'norm_y_{i}', f'norm_z_{i}'])
        
    X_coords = df[norm_cols].values
    y = df['class_id'].values
    
    X_features = []
    for row in X_coords:
        landmarks = row.reshape(21, 3)
        
        # 1. 15 Joint Angles
        angles = []
        finger_triples = [
            # Thumb
            (0, 1, 2), (1, 2, 3), (2, 3, 4),
            # Index
            (0, 5, 6), (5, 6, 7), (6, 7, 8),
            # Middle
            (0, 9, 10), (9, 10, 11), (10, 11, 12),
            # Ring
            (0, 13, 14), (13, 14, 15), (14, 15, 16),
            # Pinky
            (0, 17, 18), (17, 18, 19), (18, 19, 20)
        ]
        for p1, p2, p3 in finger_triples:
            v1 = landmarks[p1] - landmarks[p2]
            v2 = landmarks[p3] - landmarks[p2]
            dot = np.dot(v1, v2)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                ang = 0.0
            else:
                cos_theta = np.clip(dot / (n1 * n2), -1.0, 1.0)
                ang = np.arccos(cos_theta)
            angles.append(ang)
            
        # 2. 10 fingertip distances (already normalized since coordinates are normalized)
        tips = [4, 8, 12, 16, 20]
        distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                t1 = landmarks[tips[i]]
                t2 = landmarks[tips[j]]
                distances.append(np.linalg.norm(t1 - t2))
                
        geom_features = angles + distances
        
        if include_coords:
            # Combine 25 geometric features + 63 normalized coordinates
            row_features = np.concatenate([geom_features, row])
            X_features.append(row_features)
        else:
            X_features.append(geom_features)
            
    return np.array(X_features), y

def train_model(output_dir="./data"):
    # Load class mapping
    mapping_path = os.path.join(output_dir, "class_mapping.json")
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    class_names = [class_mapping[str(i)] for i in range(num_classes)]
    
    print("Extracting features from train and validation sets...")
    X_train, y_train = load_and_extract_features(os.path.join(output_dir, "train.csv"), include_coords=True)
    X_val, y_val = load_and_extract_features(os.path.join(output_dir, "valid.csv"), include_coords=True)
    
    input_size = X_train.shape[1]
    print(f"Dataset Loaded. Feature vector size: {input_size}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # PyTorch Data Loaders
    train_dataset = HandSignsDataset(X_train, y_train)
    val_dataset = HandSignsDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Device selection: Apple Silicon GPU (mps) -> NVIDIA GPU (cuda) -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using training device: {device}")
    
    # Initialize network
    model = NarutoMLP(input_size=input_size, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training Loop params
    num_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train
        
        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_batch).sum().item()
                total_val += y_batch.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{num_epochs} - "
              f"Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc*100:.2f}%")
              
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "mlp_model.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Model restored to best weights.")
                break
                
    # 3. Load best weights and evaluate
    model.load_state_dict(torch.load(os.path.join(output_dir, "mlp_model.pth")))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            
    print("\n--- Final Validation Metrics ---")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # 4. Save Plots
    # Loss & Accuracy Curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_accuracy.png"))
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    print(f"Model saved to: {os.path.join(output_dir, 'mlp_model.pth')}")
    print(f"Training charts saved to: {output_dir}")

if __name__ == "__main__":
    train_model()
