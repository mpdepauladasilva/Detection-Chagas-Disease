#!/usr/bin/env python
"""
ECG Preprocessing Pipeline for CSV-based Dataset
PhysioNet Challenge 2025 - Stable & Test-Safe Version

✔ No augmentation
✔ No data leakage
✔ Robust normalization
✔ Correct path handling
✔ WFDB-safe loading
✔ Energy sanity check
"""

import os
import numpy as np
import pandas as pd
import json
from scipy.signal import resample_poly, iirnotch, sosfiltfilt, butter, filtfilt
from math import gcd
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

class PreprocessingConfig:

    BASE_DATA_PATH = "/media/mpsilva-lx/Dados/Recursos"

    TARGET_SAMPLING_RATE = 400
    ECG_LENGTH = 4096

    BANDPASS_LOW = 0.5
    BANDPASS_HIGH = 45.0

    POWERLINE_FREQS = [50, 60]
    NOTCH_Q_FACTOR = 30.0

    EDGE_PAD_SECONDS = 3
    NORMALIZATION_METHOD = "robust"

    SAVE_FORMAT = "numpy"
    TEST_RECORDS = 5


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def resample_signal(signal, original_fs, target_fs):
    if original_fs == target_fs:
        return signal

    g = gcd(int(target_fs), int(original_fs))
    up = target_fs // g
    down = original_fs // g

    return np.array([
        resample_poly(lead, up, down)
        for lead in signal
    ])


def remove_powerline_interference(signal, fs, freqs, q):
    x = signal.copy()
    for f in freqs:
        b, a = iirnotch(f, q, fs=fs)
        x = filtfilt(b, a, x, axis=-1)
    return x


def bandpass_filter(signal, fs, low, high, order=4):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, signal, axis=-1)


def normalize_signal(signal, method="robust"):
    x = signal.copy()

    for i in range(x.shape[0]):
        lead = x[i]

        p1, p99 = np.percentile(lead, [1, 99])
        lead = np.clip(lead, p1, p99)

        if method == "robust":
            median = np.median(lead)
            mad = np.median(np.abs(lead - median))
            mad = mad if mad > 1e-8 else 1.0
            lead = (lead - median) / (1.4826 * mad)

        elif method == "zscore":
            std = np.std(lead)
            std = std if std > 1e-8 else 1.0
            lead = (lead - np.mean(lead)) / std

        lead = np.clip(lead, -5, 5)
        x[i] = lead

    return x


def standardize_length(signal, target_length):
    leads, length = signal.shape

    if length < target_length:
        pad = target_length - length
        signal = np.pad(signal, ((0, 0), (0, pad)), mode="edge")

    elif length > target_length:
        start = (length - target_length) // 2
        signal = signal[:, start:start + target_length]

    return signal


# ============================================================================
# HEADER PARSING
# ============================================================================

def get_sampling_frequency(header):
    for line in header.split("\n"):
        if line.strip().startswith("0 "):  # WFDB first header line
            parts = line.split()
            if len(parts) >= 3:
                try:
                    return int(parts[2])
                except:
                    pass
    return None


# ============================================================================
# WFDB LOADING (ROBUST)
# ============================================================================

def load_signals(record_path):
    import wfdb

    record_path = record_path.replace(".hea", "").replace(".dat", "")
    record = wfdb.rdrecord(record_path)

    header_path = record_path + ".hea"
    with open(header_path, "r") as f:
        header = f.read()

    return record.p_signal.T, header


# ============================================================================
# PREPROCESS CORE
# ============================================================================

def preprocess_ecg_test(signal, header, config):

    signal = np.nan_to_num(signal)

    fs = get_sampling_frequency(header)
    if fs is None:
        fs = 500

    # Resample
    signal = resample_signal(signal, fs, config.TARGET_SAMPLING_RATE)

    # Reflection padding
    pad = int(config.EDGE_PAD_SECONDS * config.TARGET_SAMPLING_RATE)
    signal = np.pad(signal, ((0, 0), (pad, pad)), mode="reflect")

    # Notch
    signal = remove_powerline_interference(
        signal,
        config.TARGET_SAMPLING_RATE,
        config.POWERLINE_FREQS,
        config.NOTCH_Q_FACTOR
    )

    # Bandpass
    signal = bandpass_filter(
        signal,
        config.TARGET_SAMPLING_RATE,
        config.BANDPASS_LOW,
        config.BANDPASS_HIGH
    )

    signal = signal[:, pad:-pad]
    signal = standardize_length(signal, config.ECG_LENGTH)
    signal = normalize_signal(signal, config.NORMALIZATION_METHOD)

    energy = np.mean(signal**2)
    if energy < 1e-6:
        raise ValueError("Signal energy collapsed after preprocessing.")

    return signal.astype(np.float32)


# ============================================================================
# CSV PIPELINE
# ============================================================================

def preprocess_record_from_csv(row, config):

    record_id = row["record_id"]
    file_path = str(row["file_path"]).strip()

    # Remove leading slash (critical fix)
    file_path = file_path.lstrip("/")

    # Remove extension
    file_path = file_path.replace(".hea", "").replace(".dat", "")

    full_path = os.path.abspath(
        os.path.join(config.BASE_DATA_PATH, file_path)
    )

    if not os.path.exists(full_path + ".hea"):
        print(f"File not found: {full_path}")
        return None, None

    try:
        signal, header = load_signals(full_path)
    except Exception as e:
        print(f"Load error {record_id}: {e}")
        return None, None

    processed = preprocess_ecg_test(signal, header, config)

    metadata = {
        "record_id": record_id,
        "dataset": row["dataset"],
        "label": int(row["chagas_label"]),
        "original_shape": signal.shape,
        "processed_shape": processed.shape,
    }

    return processed, metadata


def save_preprocessed(signal, metadata, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, signal=signal, metadata=metadata)


def preprocess_csv_dataset(csv_path, output_folder, config):

    start = datetime.now()
    df = pd.read_csv(csv_path, low_memory=False).head(config.TEST_RECORDS)

    stats = {"success": 0, "errors": 0}

    for _, row in df.iterrows():
        record_id = row["record_id"]
        print(f"Processing {record_id}...")

        signal, metadata = preprocess_record_from_csv(row, config)

        if signal is None:
            stats["errors"] += 1
            continue

        out_path = os.path.join(output_folder, f"{record_id}_processed.npz")
        save_preprocessed(signal, metadata, out_path)

        stats["success"] += 1
        print(f"  ✔ Shape {signal.shape}")

    stats["duration_sec"] = (datetime.now() - start).total_seconds()

    with open(os.path.join(output_folder, "preprocessing_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nFinished.")
    print(stats)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    CSV_PATH = "/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/dataset_chagas_absolut.csv"
    OUTPUT_FOLDER = "./data/preprocessed"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    config = PreprocessingConfig()

    preprocess_csv_dataset(
        csv_path=CSV_PATH,
        output_folder=OUTPUT_FOLDER,
        config=config
    )