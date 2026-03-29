"""
model_ecg.py
------------
1D CNN for ECG arrhythmia detection (WECARE).
Input:  (batch, 1, 256)  — single-lead beat segment
Output: (batch, 2)       — logits for [Normal, Arrhythmia]

Results on MIT-BIH test set:
    Accuracy  = 0.9933
    Precision = 0.9834
    Recall    = 0.9894
    F1-Score  = 0.9864
    AUC-ROC   = 0.9992
    Latency   = 0.018 ms/beat

Author: Tarun Sadarla
"""

import torch
import torch.nn as nn


class ECG_CNN(nn.Module):
    """
    3-block 1D CNN for beat-level ECG arrhythmia classification.
    TorchScript-compatible for edge deployment.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2):
        super().__init__()
        self.conv_net = nn.Sequential(
            # Block 1 — fine QRS morphology (256 → 128)
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2 — P/T wave context (128 → 64)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3 — high-level rhythm features (64 → 32)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_net(x))


def get_model(device: torch.device) -> ECG_CNN:
    return ECG_CNN(in_channels=1, n_classes=2).to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    dummy = torch.randn(8, 1, 256).to(device)
    print(f"Output shape: {model(dummy).shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
