"""
preprocess_ecg.py
-----------------
ECG signal preprocessing for WECARE arrhythmia detection.
Processes the MIT-BIH Arrhythmia Database (PhysioNet).

Pipeline:
    1. Load WFDB records and beat annotations
    2. Bandpass filter (0.5–40 Hz Butterworth) to remove baseline wander
    3. R-peak detection and beat segmentation (256-sample windows)
    4. Binary labeling: Normal (0) vs. Arrhythmia (1)
    5. StandardScaler normalization per segment

Author: Tarun Sadarla
"""

import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

# MIT-BIH arrhythmia beat annotation symbols
ARRHYTHMIA_CODES = ['V', 'A', 'L', 'R', '!', 'E', 'e', 'f', 'J', 'j', 'S', 'a', 'F']
SEGMENT_LENGTH = 256   # samples per beat (centered on R-peak)
FS = 360               # MIT-BIH sampling rate in Hz


def bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0,
                    fs: int = FS, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter to remove baseline wander and HF noise."""
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def load_record(record_path: str) -> tuple:
    """Load a WFDB record; return signal, R-peak positions, beat symbols, and fs."""
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]       # channel 0 (MLII lead)
    return signal, annotation.sample, annotation.symbol, record.fs


def segment_beats(signal: np.ndarray, r_peaks: np.ndarray, symbols: list,
                  seg_len: int = SEGMENT_LENGTH) -> tuple:
    """
    Segment signal into fixed-length windows centered on each R-peak.
    Drops beats too close to signal boundaries.
    Returns: X (n_beats, seg_len), y (n_beats,)
    """
    half = seg_len // 2
    X, y = [], []
    for peak, sym in zip(r_peaks, symbols):
        start, end = peak - half, peak + half
        if start < 0 or end > len(signal):
            continue
        X.append(signal[start:end])
        y.append(1 if sym in ARRHYTHMIA_CODES else 0)
    return np.array(X), np.array(y)


def preprocess_dataset(data_dir: str, record_ids: list) -> tuple:
    """
    Full preprocessing pipeline across a list of MIT-BIH record IDs.
    Returns normalized beat segments (X), binary labels (y), and fitted scaler.
    """
    all_X, all_y = [], []
    for rid in record_ids:
        try:
            signal, r_peaks, symbols, fs = load_record(os.path.join(data_dir, str(rid)))
            filtered = bandpass_filter(signal, fs=fs)
            X, y = segment_beats(filtered, r_peaks, symbols)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"  Skipping record {rid}: {e}")

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    print(f"Total segments: {len(X_all):,} | Normal: {(y_all==0).sum():,} | Arrhythmia: {(y_all==1).sum():,}")
    return X_all, y_all, scaler


if __name__ == "__main__":
    DATA_DIR = "data/mitbih"
    RECORD_IDS = list(range(100, 235))
    X, y, scaler = preprocess_dataset(DATA_DIR, RECORD_IDS)
    print(f"X shape: {X.shape} | y shape: {y.shape}")
