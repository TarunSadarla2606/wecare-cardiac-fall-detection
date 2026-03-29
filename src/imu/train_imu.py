"""
train_imu.py
------------
Training loop for the WECARE IMU fall detection model.
Key: threshold tuned to 0.65 (higher than default 0.5) to reduce
false positives while keeping false negatives near zero.

Author: Ramyasri Murugesan
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from model_imu import get_model


def build_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                      batch_size: int = 32):
    """Convert numpy arrays to PyTorch DataLoaders. Transpose to channels-first."""

    def make_ds(X, y):
        # (N, 100, 9) → (N, 9, 100) for Conv1d
        Xt = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        return TensorDataset(Xt, yt)

    train_ds = make_ds(X_train, y_train)
    val_ds   = make_ds(X_val,   y_val)
    test_ds  = make_ds(X_test,  y_test)

    # WeightedRandomSampler to counter class imbalance
    class_counts = np.bincount(y_train)
    sample_weights = (1.0 / (class_counts + 1e-8))[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train(X_train, y_train, X_val, y_val, X_test, y_test,
          epochs: int = 20, lr: float = 1e-3, weight_decay: float = 5e-5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test)

    model = get_model(device)
    criterion = nn.CrossEntropyLoss()
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
    print("Import and call train() with outputs from preprocess_imu.build_dataset().")
