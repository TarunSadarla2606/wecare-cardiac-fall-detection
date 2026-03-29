"""
preprocess_imu.py
-----------------
IMU signal preprocessing for WECARE fall detection.
Processes the MobiFall_processed dataset (Kaggle).

Pipeline:
    1. Load CSV trial files (9 channels: acc/gyro/orientation)
    2. Forward/backward interpolation for missing samples
    3. Butterworth low-pass filtering for motion noise reduction
    4. Sliding window segmentation (100 samples × 9 channels)
    5. Label extraction from filename (Fall vs. ADL)
    6. StandardScaler normalization per channel

Fall types: FOL (forward), FKL (fall-kneel), BSC (backward-sitting), SDL (side-lying)
ADL types:  Walking, standing, sitting, stairs, etc.

Author: Ramyasri Murugesan
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FALL_CODES   = ['FOL', 'FKL', 'BSC', 'SDL']
SENSOR_COLS  = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
ORI_COLS     = ['ori_azimuth', 'ori_pitch', 'ori_roll']
ALL_COLS     = SENSOR_COLS + ORI_COLS   # 9 channels total

WINDOW_SIZE  = 100    # samples per window
STEP_SIZE    = 50     # 50% overlap
RANDOM_SEED  = 42


def is_fall(filename: str) -> int:
    """Return 1 if filename contains a fall activity code, else 0."""
    fn = os.path.basename(filename).upper()
    return 1 if any(code in fn for code in FALL_CODES) else 0


def butterworth_filter(signal: np.ndarray, cutoff: float = 20.0,
                       fs: float = 100.0, order: int = 4) -> np.ndarray:
    """Low-pass Butterworth filter to remove high-frequency motion noise."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal, axis=0)


def load_and_clean(file_path: str) -> pd.DataFrame:
    """Load CSV, ensure required columns exist, interpolate missing values."""
    df = pd.read_csv(file_path)
    df = df.ffill().bfill()   # handle minor sensor dropouts
    return df


def segment_trial(df: pd.DataFrame, window_size: int = WINDOW_SIZE,
                  step_size: int = STEP_SIZE) -> np.ndarray:
    """
    Sliding window segmentation over a trial DataFrame.
    Returns array of shape (n_windows, window_size, 9).
    """
    cols = [c for c in ALL_COLS if c in df.columns]
    arr = df[cols].values.astype(np.float32)
    segments = []
    for start in range(0, len(arr) - window_size + 1, step_size):
        segments.append(arr[start:start + window_size])
    return np.array(segments)   # (n_windows, 100, 9)


def build_dataset(data_folder: str,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.05) -> tuple:
    """
    Full preprocessing pipeline across all CSV files in data_folder.
    Returns train/val/test splits and fitted scaler.
    """
    csvs = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not csvs:
        raise RuntimeError(f"No CSV files found in {data_folder}")
    print(f"Total trial files: {len(csvs)}")

    # File-level stratified split (prevents data leakage across windows)
    labels_per_file = [is_fall(f) for f in csvs]
    train_files, temp_files = train_test_split(
        csvs, test_size=val_ratio + test_ratio,
        stratify=labels_per_file, random_state=RANDOM_SEED)
    temp_labels = [is_fall(f) for f in temp_files]
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels, random_state=RANDOM_SEED)

    def process_files(file_list):
        X_list, y_list = [], []
        for f in file_list:
            df = load_and_clean(f)
            segs = segment_trial(df)
            if len(segs) == 0:
                continue
            X_list.append(segs)
            y_list.extend([is_fall(f)] * len(segs))
        return np.concatenate(X_list), np.array(y_list)

    X_train, y_train = process_files(train_files)
    X_val,   y_val   = process_files(val_files)
    X_test,  y_test  = process_files(test_files)

    # StandardScaler fitted on training set, applied to all splits
    scaler = StandardScaler()
    n_train, ws, nc = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, nc)).reshape(n_train, ws, nc)
    X_val   = scaler.transform(X_val.reshape(-1, nc)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, nc)).reshape(X_test.shape)

    print(f"Train: {len(X_train):,} windows | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Fall ratio — Train: {y_train.mean():.2%} | Val: {y_val.mean():.2%} | Test: {y_test.mean():.2%}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, test_files


if __name__ == "__main__":
    DATA_DIR = "data/mobifall"
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), scaler, _ = build_dataset(DATA_DIR)
    print(f"X_train shape: {X_tr.shape}")   # (N, 100, 9)
