"""
evaluate_imu.py
---------------
Evaluation utilities for the WECARE IMU fall detection model.
Includes trial-level visualization (acceleration magnitude + fall probability)
and inference latency benchmarking.

Author: Ramyasri Murugesan
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)


THRESHOLD = 0.65    # tuned to balance precision and recall for safety


def evaluate(model, loader, device, threshold: float = THRESHOLD):
    model.eval()
    preds, targets, probs_all = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (prob >= threshold).astype(int)
            preds.extend(pred)
            targets.extend(yb.numpy())
            probs_all.extend(prob)
    return np.array(preds), np.array(targets), np.array(probs_all)


def print_metrics(preds, targets):
    print(f"  Accuracy : {accuracy_score(targets, preds):.4f}")
    print(f"  Precision: {precision_score(targets, preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(targets, preds, zero_division=0):.4f}")
    print(f"  F1-Score : {f1_score(targets, preds, zero_division=0):.4f}")
    cm = confusion_matrix(targets, preds)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")


def plot_confusion_matrix(preds, targets, save_path: str = None):
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["ADL (0)", "Fall (1)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix (Threshold={THRESHOLD})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curve(probs, targets, save_path: str = None):
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def benchmark_latency(model, loader, device, n_warmup: int = 10, n_measure: int = 100):
    """Measure per-window inference latency in milliseconds."""
    model.eval()
    data_iter = iter(loader)
    for _ in range(n_warmup):
        try:
            xb, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, _ = next(data_iter)
        with torch.no_grad():
            _ = model(xb.to(device))

    data_iter = iter(loader)
    start = time.time()
    total = 0
    for _ in range(n_measure):
        try:
            xb, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, _ = next(data_iter)
        with torch.no_grad():
            _ = model(xb.to(device))
        total += len(xb)

    elapsed_ms = (time.time() - start) * 1000
    per_window_ms = elapsed_ms / total
    print(f"  Per-window latency: {per_window_ms:.4f} ms  (target ≤ 40 ms)")
    return per_window_ms
