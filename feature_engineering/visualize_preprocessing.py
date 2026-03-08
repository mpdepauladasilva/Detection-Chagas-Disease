#!/usr/bin/env python
"""
ECG RAW vs PREPROCESSED VALIDATION
Scientifically Correct Version (Aligned + Robust Metrics)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, resample, correlate
from scipy.stats import pearsonr
import pandas as pd

from preprocess_ecg_csv import (
    load_signals,
    get_sampling_frequency,
    PreprocessingConfig
)

# ============================================================
# CONFIG
# ============================================================

PREPROCESSED_FOLDER = "./data/preprocessed"
OUTPUT_FOLDER = "./visualizations"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DISPLAY_LEADS = [0, 1, 6, 10]
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


# ============================================================
# UTILITIES
# ============================================================

def load_preprocessed_signal(path):
    data = np.load(path, allow_pickle=True)
    return data["signal"], data["metadata"].item()


def align_signals(raw, proc):
    """
    Align signals using cross-correlation.
    """
    corr = correlate(proc, raw, mode="full")
    shift = np.argmax(corr) - len(raw) + 1

    if shift > 0:
        raw = raw[shift:]
        proc = proc[:len(raw)]
    elif shift < 0:
        raw = raw[:len(proc)+shift]
        proc = proc[-shift:]

    min_len = min(len(raw), len(proc))
    return raw[:min_len], proc[:min_len]


def compute_band_energy(freqs, psd, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return float(np.trapz(psd[idx], freqs[idx]))


def compute_frequency_metrics(signal, fs):

    freqs, psd = welch(signal, fs=fs, nperseg=1024)

    baseline = compute_band_energy(freqs, psd, 0, 0.5)
    physio = compute_band_energy(freqs, psd, 0.5, 40)
    highfreq = compute_band_energy(freqs, psd, 40, 100)

    return {
        "baseline_energy": baseline,
        "physio_energy": physio,
        "highfreq_energy": highfreq,
        "baseline_ratio": baseline / (physio + 1e-8),
        "highfreq_ratio": highfreq / (physio + 1e-8),
    }


def compute_time_metrics(signal):
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "ptp": float(np.ptp(signal))
    }


def compute_band_snr(signal, fs):
    """
    SNR defined as:
    Physiological band energy (0.5–40 Hz)
    divided by
    High frequency energy (40–100 Hz)
    """
    freqs, psd = welch(signal, fs=fs, nperseg=1024)

    physio = compute_band_energy(freqs, psd, 0.5, 40)
    noise = compute_band_energy(freqs, psd, 40, 100)

    return float(physio / (noise + 1e-8))


# ============================================================
# VALIDATION CORE
# ============================================================

def validate_record(record_id, file_path, config):

    print(f"\nValidating record: {record_id}")

    file_path = str(file_path).lstrip("/")
    raw_path = os.path.join(config.BASE_DATA_PATH, file_path)
    raw_path = raw_path.replace(".hea", "").replace(".dat", "")

    signal_raw, header = load_signals(raw_path)

    fs_raw = get_sampling_frequency(header)
    if fs_raw is None:
        fs_raw = 500

    processed_path = os.path.join(
        PREPROCESSED_FOLDER,
        f"{record_id}_processed.npz"
    )

    if not os.path.exists(processed_path):
        print("Processed file not found.")
        return

    signal_proc, metadata = load_preprocessed_signal(processed_path)
    fs_proc = config.TARGET_SAMPLING_RATE

    results = {}

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle(f"ECG Validation - Record {record_id}", fontsize=18)

    for i, lead in enumerate(DISPLAY_LEADS):

        raw = signal_raw[lead]

        # Resample RAW to match processed sampling rate
        raw_resampled = resample(raw, len(signal_proc[lead]))

        proc = signal_proc[lead]

        # Align signals
        raw_aligned, proc_aligned = align_signals(raw_resampled, proc)

        # -----------------------
        # Metrics
        # -----------------------
        time_raw = compute_time_metrics(raw_aligned)
        time_proc = compute_time_metrics(proc_aligned)

        freq_raw = compute_frequency_metrics(raw_aligned, fs_proc)
        freq_proc = compute_frequency_metrics(proc_aligned, fs_proc)

        snr_raw = compute_band_snr(raw_aligned, fs_proc)
        snr_proc = compute_band_snr(proc_aligned, fs_proc)

        corr = pearsonr(raw_aligned, proc_aligned)[0]

        results[LEAD_NAMES[lead]] = {
            "time_raw": time_raw,
            "time_processed": time_proc,
            "freq_raw": freq_raw,
            "freq_processed": freq_proc,
            "snr_raw_band": snr_raw,
            "snr_processed_band": snr_proc,
            "correlation_morphology": float(corr)
        }

        # -----------------------
        # Plot
        # -----------------------
        t = np.arange(len(proc_aligned)) / fs_proc

        axes[i, 0].plot(t[:2000], raw_aligned[:2000])
        axes[i, 0].set_title(f"{LEAD_NAMES[lead]} - RAW (aligned)")

        axes[i, 1].plot(t[:2000], proc_aligned[:2000])
        axes[i, 1].set_title(f"{LEAD_NAMES[lead]} - PROCESSED")

        f1, p1 = welch(raw_aligned, fs=fs_proc, nperseg=1024)
        f2, p2 = welch(proc_aligned, fs=fs_proc, nperseg=1024)

        axes[i, 2].semilogy(f1, p1, label="RAW")
        axes[i, 2].semilogy(f2, p2, label="PROCESSED")
        axes[i, 2].set_xlim(0, 100)
        axes[i, 2].legend()
        axes[i, 2].set_title(f"{LEAD_NAMES[lead]} - PSD")

    plt.tight_layout()

    fig_path = os.path.join(
        OUTPUT_FOLDER,
        f"{record_id}_validation_corrected.png"
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()

    metrics_path = os.path.join(
        OUTPUT_FOLDER,
        f"{record_id}_metrics_corrected.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Validation saved.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    config = PreprocessingConfig()

    CSV_PATH = "/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/dataset_chagas_absolut.csv"

    RECORDS = [
        265341,
        483197,
        356282,
        1411959,
        1628260
    ]

    df = pd.read_csv(CSV_PATH)

    for record_id in RECORDS:
        row = df[df["record_id"] == record_id].iloc[0]

        validate_record(
            record_id=record_id,
            file_path=row["file_path"],
            config=config
        )

    print("\n✅ Scientifically corrected validation complete.")