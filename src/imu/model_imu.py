"""
model_imu.py
------------
1D CNN for IMU-based fall detection (WECARE).
Input:  (batch, 9, 100) — 9-channel IMU window (channels-first for Conv1d)
Output: (batch, 2)      — logits for [ADL, Fall]

Results on MobiFall test set (threshold=0.65):
    Accuracy  = 0.9178
    Precision = 0.7029
    Recall    = 0.9649   ← critical safety metric (FN = 8)
    F1-Score  = 0.8133
    AUC-ROC   = 0.9810
    Latency   = 0.033 ms/window

Design note: Dropout is intentionally kept low (0.2) to preserve
Recall — in fall detection, a missed fall is far more dangerous
than a false alarm.

Author: Ramyasri Murugesan
"""

import torch
import torch.nn as nn


class IMU_CNN(nn.Module):
    """
    3-block 1D CNN for window-level IMU fall classification.
    TorchScript-compatible for edge deployment.
    """

    def __init__(self, in_channels: int = 9, n_classes: int = 2):
        super().__init__()
        self.conv_net = nn.Sequential(
            # Block 1 — capture impact spike patterns (100 → 50)
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2 — broader fall trajectory (50 → 25)
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3 — high-level fall signature (25 → 12)
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 9, 100)
        return self.classifier(self.conv_net(x))


def get_model(device: torch.device) -> IMU_CNN:
    return IMU_CNN(in_channels=9, n_classes=2).to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    dummy = torch.randn(8, 9, 100).to(device)
    print(f"Output shape: {model(dummy).shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
