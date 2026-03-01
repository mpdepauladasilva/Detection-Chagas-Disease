#!/usr/bin/env python
"""
ECG RAW vs PREPROCESSED VALIDATION
Visual + Quantitative Metrics (Paper-Ready)

Author: ECG Validation Pipeline
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr

from preprocess_ecg_csv import (
    load_signals,
    load_preprocessed_signal,
    get_sampling_frequency,
    PreprocessingConfig
)

# ======================================================================
# CONFIG
# ======================================================================

PREPROCESSED_FOLDER = "/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/data/preprocessed"
OUTPUT_FOLDER = "/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/visualizations"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DISPLAY_LEADS = [0, 1, 6, 10]  # I, II, V1, V5
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


# ======================================================================
# METRICS
# ======================================================================

def compute_time_metrics(signal):
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "ptp": float(np.ptp(signal))
    }


def estimate_snr(signal):
    noise = np.diff(signal)
    return float(np.std(signal) / (np.std(noise) + 1e-8))


def compute_band_energy(freqs, psd, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return float(np.trapz(psd[idx], freqs[idx]))


def compute_frequency_metrics(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)

    return {
        "baseline_energy": compute_band_energy(freqs, psd, 0, 0.5),
        "physiological_energy": compute_band_energy(freqs, psd, 0.5, 40),
        "highfreq_noise_energy": compute_band_energy(freqs, psd, 40, 100),
        "powerline_50hz": compute_band_energy(freqs, psd, 49, 51),
        "powerline_60hz": compute_band_energy(freqs, psd, 59, 61),
    }


# ======================================================================
# VALIDATION CORE
# ======================================================================

def validate_record(record_id, file_path, config):

    print(f"\nValidating record: {record_id}")

    # --------------------------------------------------
    # Load RAW
    # --------------------------------------------------
    if file_path.startswith('/'):
        file_path = file_path[1:]

    raw_path = os.path.join(config.BASE_DATA_PATH, file_path)

    signal_raw, header = load_signals(raw_path)
    fs_raw = get_sampling_frequency(header)
    if fs_raw is None:
        fs_raw = 500

    # --------------------------------------------------
    # Load PROCESSED (saved .npz)
    # --------------------------------------------------
    processed_path = os.path.join(
        PREPROCESSED_FOLDER,
        f"{record_id}_processed.npz"
    )

    signal_proc, metadata = load_preprocessed_signal(processed_path, config)
    fs_proc = config.TARGET_SAMPLING_RATE

    # --------------------------------------------------
    # Compute metrics per lead
    # --------------------------------------------------
    results = {}

    for lead in DISPLAY_LEADS:
        raw = signal_raw[lead]
        proc = signal_proc[lead]

        # Align length if needed
        min_len = min(len(raw), len(proc))
        raw = raw[:min_len]
        proc = proc[:min_len]

        time_metrics_raw = compute_time_metrics(raw)
        time_metrics_proc = compute_time_metrics(proc)

        freq_metrics_raw = compute_frequency_metrics(raw, fs_raw)
        freq_metrics_proc = compute_frequency_metrics(proc, fs_proc)

        corr = pearsonr(raw[:len(proc)], proc)[0]

        results[LEAD_NAMES[lead]] = {
            "raw_time": time_metrics_raw,
            "processed_time": time_metrics_proc,
            "raw_freq": freq_metrics_raw,
            "processed_freq": freq_metrics_proc,
            "snr_raw": estimate_snr(raw),
            "snr_processed": estimate_snr(proc),
            "correlation_raw_processed": float(corr)
        }

    # --------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------
    metrics_path = os.path.join(
        OUTPUT_FOLDER,
        f"{record_id}_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Metrics saved to {metrics_path}")

    # --------------------------------------------------
    # Generate Figure
    # --------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle(f"ECG Validation - Record {record_id}", fontsize=18)

    for i, lead in enumerate(DISPLAY_LEADS):
        raw = signal_raw[lead]
        proc = signal_proc[lead]

        t_raw = np.arange(len(raw)) / fs_raw
        t_proc = np.arange(len(proc)) / fs_proc

        # Raw
        axes[i, 0].plot(t_raw[:2000], raw[:2000])
        axes[i, 0].set_title(f"{LEAD_NAMES[lead]} - RAW")

        # Processed
        axes[i, 1].plot(t_proc[:2000], proc[:2000])
        axes[i, 1].set_title(f"{LEAD_NAMES[lead]} - PROCESSED")

        # PSD overlay
        f1, p1 = welch(raw, fs=fs_raw, nperseg=1024)
        f2, p2 = welch(proc, fs=fs_proc, nperseg=1024)

        axes[i, 2].semilogy(f1, p1, label="RAW")
        axes[i, 2].semilogy(f2, p2, label="PROCESSED")
        axes[i, 2].set_xlim(0, 100)
        axes[i, 2].legend()
        axes[i, 2].set_title(f"{LEAD_NAMES[lead]} - PSD")

    plt.tight_layout()
    fig_path = os.path.join(
        OUTPUT_FOLDER,
        f"{record_id}_validation.png"
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Figure saved to {fig_path}")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    config = PreprocessingConfig()

    RECORDS = [
        265341,
        483197,
        356282,
        1411959,
        1628260
    ]

    CSV_PATH = "/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/dataset_chagas_absolut.csv"

    import pandas as pd
    df = pd.read_csv(CSV_PATH)

    for record_id in RECORDS:
        row = df[df["record_id"] == record_id].iloc[0]
        validate_record(
            record_id=record_id,
            file_path=row["file_path"],
            config=config
        )

    print("\n✅ Full validation complete.")