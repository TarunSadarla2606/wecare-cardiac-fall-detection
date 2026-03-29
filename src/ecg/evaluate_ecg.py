"""
evaluate_ecg.py
---------------
Evaluation utilities for the WECARE ECG arrhythmia detection model.
Produces: confusion matrix, ROC curve, F1/precision/recall,
          single-beat visualization, and inference latency benchmark.

Author: Tarun Sadarla
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)


def evaluate(model, loader, device, threshold: float = 0.5):
    """Run inference on a DataLoader; return predictions and targets."""
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
    print(f"  Confusion Matrix:\n{confusion_matrix(targets, preds)}")


def plot_confusion_matrix(preds, targets, save_path: str = None):
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal (0)", "Arrhythmia (1)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix (Test Set)")
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
    """Measure per-beat inference latency in milliseconds."""
    model.eval()
    data_iter = iter(loader)

    # Warm-up
    for _ in range(n_warmup):
        try:
            xb, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, _ = next(data_iter)
        with torch.no_grad():
            _ = model(xb.to(device))

    # Measure
    data_iter = iter(loader)
    start = time.time()
    total_samples = 0
    for _ in range(n_measure):
        try:
            xb, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, _ = next(data_iter)
        with torch.no_grad():
            _ = model(xb.to(device))
        total_samples += len(xb)

    elapsed_ms = (time.time() - start) * 1000
    per_sample_ms = elapsed_ms / total_samples
    print(f"  Total inference time : {elapsed_ms:.2f} ms over {total_samples} beats")
    print(f"  Per-beat latency     : {per_sample_ms:.4f} ms")
    return per_sample_ms
