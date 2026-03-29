"""
train_ecg.py
------------
Training loop for the WECARE ECG arrhythmia detection model.
Handles severe class imbalance (~75% Normal) using:
  - Class weights passed to CrossEntropyLoss
  - WeightedRandomSampler for balanced batch construction

Author: Tarun Sadarla
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from model_ecg import get_model


def build_dataloaders(X: np.ndarray, y: np.ndarray,
                      val_ratio: float = 0.10,
                      test_ratio: float = 0.10,
                      batch_size: int = 64,
                      random_seed: int = 42):
    """Stratified split + weighted sampler for training set."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=random_seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio / (1 - test_ratio),
        stratify=y_temp, random_state=random_seed)

    # Reshape for Conv1d: (N, 1, 256)
    def to_tensor(arr_x, arr_y):
        X_t = torch.tensor(arr_x, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(arr_y, dtype=torch.long)
        return TensorDataset(X_t, y_t)

    train_ds = to_tensor(X_train, y_train)
    val_ds   = to_tensor(X_val,   y_val)
    test_ds  = to_tensor(X_test,  y_test)

    # WeightedRandomSampler for class imbalance
    class_counts = np.bincount(y_train)
    weights = 1.0 / (class_counts + 1e-8)
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_counts


def train(X: np.ndarray, y: np.ndarray,
          epochs: int = 30,
          lr: float = 1e-3,
          weight_decay: float = 5e-5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, val_loader, test_loader, class_counts = build_dataloaders(X, y)

    model = get_model(device)

    # Class-weighted loss to further penalize missed arrhythmias
    class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(device)).argmax(dim=1).cpu()
                correct += (preds == yb).sum().item()
                total += len(yb)

        print(f"Epoch {epoch:02d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {correct/total:.4f}")

    return model, test_loader


if __name__ == "__main__":
    # Load your preprocessed data here
    # X, y, _ = preprocess_dataset("data/mitbih", list(range(100, 235)))
    # model, test_loader = train(X, y, epochs=30)
    print("Import and call train(X, y) with preprocessed data.")
