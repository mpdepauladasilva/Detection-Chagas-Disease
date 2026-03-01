#!/usr/bin/env python
"""
Visualize ECG signals before and after preprocessing
Compare raw vs processed signals for quality control

Based on team_code.py from George B. Moody PhysioNet Challenge 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
import sys

# Import preprocessing functions
from preprocess_ecg_csv import (
    load_signals, 
    load_header, 
    get_sampling_frequency,
    preprocess_ecg_test,
    PreprocessingConfig
)


def plot_signal_comparison(record_id, file_path, config, save_path=None):
    """
    Plot raw vs preprocessed ECG signal.
    
    Args:
        record_id: Record identifier
        file_path: Path to .hea file (relative to BASE_DATA_PATH)
        config: PreprocessingConfig object
        save_path: Path to save the figure (optional)
    """
    print(f"Loading record: {record_id}")
    
    # ========================================================================
    # CONCATENATE BASE PATH WITH CSV PATH (same as preprocess_ecg_csv.py)
    # ========================================================================
    if file_path.startswith('/'):
        file_path = file_path[1:]
    
    full_file_path = os.path.join(config.BASE_DATA_PATH, file_path)
    
    if not os.path.exists(full_file_path):
        print(f"ERROR: File not found at {full_file_path}")
        return None, None
    # ========================================================================
    
    print(f"  Full path: {full_file_path}")
    
    # Load raw signal
    try:
        signal_raw, header = load_signals(full_file_path)
    except Exception as e:
        print(f"ERROR loading signal: {e}")
        return None, None
    
    fs_raw = get_sampling_frequency(header)
    if fs_raw is None:
        fs_raw = 500
    
    # Get label from header or assume 0
    label = 0
    
    # Create metadata row for preprocessing
    class Row:
        def __init__(self, record_id, file_path, label):
            self.record_id = record_id
            self.file_path = file_path
            self.chagas_label = label
            self.dataset = 'CODE-15%'
            self.age = 'Unknown'
            self.sex = 'Unknown'
        
        def get(self, key, default='Unknown'):
            return getattr(self, key, default)
    
    row = Row(record_id, file_path, label)
    
    # Preprocess
    signal_processed, metadata = preprocess_record_from_csv(row, config)
    
    if signal_processed is None:
        print("Error: Could not preprocess signal")
        return None, None
    
    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle(f'ECG Preprocessing Comparison - Record {record_id}', fontsize=16, fontweight='bold')
    
    # Lead names
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Select 4 leads to display (I, II, V1, V5)
    display_leads = [0, 1, 6, 10]
    
    # Time axes
    time_raw = np.arange(signal_raw.shape[1]) / fs_raw
    time_proc = np.arange(signal_processed.shape[1]) / config.TARGET_SAMPLING_RATE
    
    # Row 1: Lead I - Raw vs Processed (time domain)
    lead_idx = display_leads[0]
    axes[0, 0].plot(time_raw[:2000], signal_raw[lead_idx, :2000], 'blue', linewidth=0.5, label='Raw')
    axes[0, 0].set_title(f'Lead {lead_names[lead_idx]} - Raw Signal', fontsize=12)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(time_proc[:2000], signal_processed[lead_idx, :2000], 'green', linewidth=0.5, label='Processed')
    axes[0, 1].set_title(f'Lead {lead_names[lead_idx]} - Processed Signal', fontsize=12)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude (z-score)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Row 2: Lead II - Raw vs Processed (time domain)
    lead_idx = display_leads[1]
    axes[1, 0].plot(time_raw[:2000], signal_raw[lead_idx, :2000], 'blue', linewidth=0.5)
    axes[1, 0].set_title(f'Lead {lead_names[lead_idx]} - Raw Signal', fontsize=12)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude (μV)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_proc[:2000], signal_processed[lead_idx, :2000], 'green', linewidth=0.5)
    axes[1, 1].set_title(f'Lead {lead_names[lead_idx]} - Processed Signal', fontsize=12)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude (z-score)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Row 3: Power Spectral Density (Lead I)
    lead_idx = display_leads[0]
    freq_raw, psd_raw = welch(signal_raw[lead_idx, :], fs=fs_raw, nperseg=1024)
    freq_proc, psd_proc = welch(signal_processed[lead_idx, :], fs=config.TARGET_SAMPLING_RATE, nperseg=1024)
    
    axes[2, 0].semilogy(freq_raw, psd_raw, 'blue', linewidth=0.5)
    axes[2, 0].set_title(f'Lead {lead_names[lead_idx]} - Power Spectral Density (Raw)', fontsize=12)
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('Power/Hz')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim([0, 100])
    
    axes[2, 1].semilogy(freq_proc, psd_proc, 'green', linewidth=0.5)
    axes[2, 1].set_title(f'Lead {lead_names[lead_idx]} - Power Spectral Density (Processed)', fontsize=12)
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Power/Hz')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlim([0, 100])
    
    # Row 4: Statistics comparison
    stats_raw = {
        'Mean': np.mean(signal_raw),
        'Std': np.std(signal_raw),
        'Min': np.min(signal_raw),
        'Max': np.max(signal_raw),
        'SNR (est)': np.std(signal_raw) / max(np.std(np.diff(signal_raw)), 1e-6)
    }
    
    stats_proc = {
        'Mean': np.mean(signal_processed),
        'Std': np.std(signal_processed),
        'Min': np.min(signal_processed),
        'Max': np.max(signal_processed),
        'SNR (est)': np.std(signal_processed) / max(np.std(np.diff(signal_processed)), 1e-6)
    }
    
    axes[3, 0].axis('off')
    axes[3, 0].text(0.1, 0.9, 'RAW SIGNAL STATISTICS', fontsize=14, fontweight='bold', transform=axes[3, 0].transAxes)
    for i, (key, value) in enumerate(stats_raw.items()):
        axes[3, 0].text(0.1, 0.8 - i*0.15, f'{key}: {value:.4f}', fontsize=11, transform=axes[3, 0].transAxes)
    
    axes[3, 1].axis('off')
    axes[3, 1].text(0.1, 0.9, 'PROCESSED SIGNAL STATISTICS', fontsize=14, fontweight='bold', transform=axes[3, 1].transAxes)
    for i, (key, value) in enumerate(stats_proc.items()):
        axes[3, 1].text(0.1, 0.8 - i*0.15, f'{key}: {value:.4f}', transform=axes[3, 1].transAxes)
    
    # Add preprocessing info
    fig.text(0.5, 0.02, 
             f'Preprocessing: Resample={config.TARGET_SAMPLING_RATE}Hz | '
             f'Bandpass={config.BANDPASS_LOW}-{config.BANDPASS_HIGH}Hz | '
             f'Notch={config.POWERLINE_FREQS}Hz | '
             f'Length={config.ECG_LENGTH} samples | '
             f'Normalization={config.NORMALIZATION_METHOD}',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return signal_raw, signal_processed


def preprocess_record_from_csv(row, config):
    """Helper function for visualization (same as in preprocess_ecg_csv.py)"""
    record_id = row.record_id
    file_path = row.file_path
    
    # Concatenate base path
    if file_path.startswith('/'):
        file_path = file_path[1:]
    
    full_file_path = os.path.join(config.BASE_DATA_PATH, file_path)
    
    if not os.path.exists(full_file_path):
        return None, None
    
    signal, header = load_signals(full_file_path)
    processed_signal = preprocess_ecg_test(signal, header, config)
    
    metadata = {
        'record_id': record_id,
        'original_fs': get_sampling_frequency(header),
        'target_fs': config.TARGET_SAMPLING_RATE,
        'original_shape': signal.shape,
        'processed_shape': processed_signal.shape
    }
    
    return processed_signal, metadata


def visualize_multiple_records(csv_path, config, num_records=5, output_folder='./visualizations'):
    """
    Visualize preprocessing for multiple records from CSV.
    
    Args:
        csv_path: Path to CSV file
        config: PreprocessingConfig object
        num_records: Number of records to visualize
        output_folder: Folder to save visualizations
    """
    df = pd.read_csv(csv_path)
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("=" * 80)
    print("ECG PREPROCESSING VISUALIZATION")
    print("=" * 80)
    print(f"CSV file: {csv_path}")
    print(f"Output folder: {output_folder}")
    print(f"BASE_DATA_PATH: {config.BASE_DATA_PATH}")
    print(f"Number of records to visualize: {num_records}")
    print("=" * 80)
    
    # Verify base path exists
    if not os.path.exists(config.BASE_DATA_PATH):
        print(f"ERROR: BASE_DATA_PATH does not exist: {config.BASE_DATA_PATH}")
        sys.exit(1)
    
    print(f"Visualizing {num_records} records...")
    print("-" * 80)
    
    success_count = 0
    error_count = 0
    
    for idx in range(min(num_records, len(df))):
        row = df.iloc[idx]
        record_id = row['record_id']
        file_path = row['file_path']
        
        save_path = os.path.join(output_folder, f'{record_id}_preprocessing_comparison.png')
        
        try:
            plot_signal_comparison(
                record_id=record_id,
                file_path=file_path,
                config=config,
                save_path=save_path
            )
            print(f"✅ [{idx + 1}/{min(num_records, len(df))}] {record_id} - Visualization saved")
            success_count += 1
        except Exception as e:
            print(f"❌ [{idx + 1}/{min(num_records, len(df))}] {record_id} - Error: {e}")
            error_count += 1
        
        print("-" * 80)
    
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"All visualizations saved to: {output_folder}")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Paths
    CSV_PATH = "./data/ecg_metadata.csv"
    OUTPUT_FOLDER = "./visualizations"
    
    # Create configuration
    config = PreprocessingConfig()
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Visualize first 5 records
    visualize_multiple_records(
        csv_path=CSV_PATH,
        config=config,
        num_records=5,
        output_folder=OUTPUT_FOLDER
    )
    
    print("\n✅ Visualization complete!")
    print(f"📁 Check the '{OUTPUT_FOLDER}' folder for before/after comparisons")