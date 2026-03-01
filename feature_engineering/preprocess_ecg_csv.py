#!/usr/bin/env python
"""
ECG Preprocessing Pipeline for CSV-based Dataset
PhysioNet Challenge 2025 - Test-Safe Preprocessing
NO augmentation, NO data leakage - suitable for train/test/val splits

Based on team_code.py from George B. Moody PhysioNet Challenge 2025
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
import joblib
from scipy.signal import resample_poly, iirnotch, sosfiltfilt, butter, filtfilt
from math import gcd
import pywt
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class PreprocessingConfig:
    """Configuration for ECG preprocessing pipeline (TEST-SAFE)"""
    
    # ========================================================================
    # PATH CONFIGURATION - MODIFY THIS FOR YOUR SYSTEM
    # ========================================================================
    BASE_DATA_PATH = "/media/mpsilva-lx/Dados/Recursos"  # Path base do seu dataset
    # ========================================================================
    
    # Resampling
    TARGET_SAMPLING_RATE = 400  # Hz (standardize all signals)
    
    # Signal length
    ECG_LENGTH = 4096  # samples (10.24s at 400Hz)
    
    # Filtering
    BANDPASS_LOW = 1.0  # Hz
    BANDPASS_HIGH = 47.0  # Hz
    POWERLINE_FREQS = [50, 60]  # Remove both 50Hz and 60Hz
    NOTCH_Q_FACTOR = 30.0  # Quality factor for notch filter
    
    # Baseline removal
    WAVELET_TYPE = 'sym8'
    WAVELET_LEVEL = 8
    
    # Normalization
    NORMALIZATION_METHOD = 'zscore'  # 'zscore' for test-safe
    
    # Edge handling
    EDGE_PAD_SECONDS = 3  # Seconds to pad for filter edge effects
    EDGE_TRIM_SAMPLES = 100  # Samples to trim from edges after filtering
    
    # Output
    SAVE_FORMAT = 'numpy'  # 'numpy' (.npz) or 'torch' (.pt)
    COMPRESSION = True
    
    # For testing - change to large number for full dataset
    TEST_RECORDS = 5  # Number of records to process for validation


# ============================================================================
# PREPROCESSING FUNCTIONS (DETERMINISTIC - NO AUGMENTATION)
# ============================================================================

def resample_signal(signal, original_fs, target_fs=400):
    """Resample ECG signal to target sampling frequency using polyphase filtering"""
    if original_fs == target_fs:
        return signal
    
    g = gcd(int(target_fs), int(original_fs))
    up = target_fs // g
    down = original_fs // g
    
    resampled = np.array([
        resample_poly(lead, up, down, axis=-1) 
        for lead in signal
    ])
    
    return resampled


def remove_powerline_interference(signal, sample_rate, powerline_freqs=[50, 60], q_factor=30.0):
    """Remove powerline interference using notch filters (deterministic)"""
    x = signal.copy()
    
    for freq in powerline_freqs:
        b, a = iirnotch(freq, q_factor, fs=sample_rate)
        x = filtfilt(b, a, x, axis=-1)
    
    return x


def remove_baseline_wavelet(signal, wavelet='sym8', level=8):
    """Remove baseline wander using wavelet decomposition (deterministic)"""
    leads, length = signal.shape
    cleaned = np.zeros_like(signal)
    
    for lead in range(leads):
        coeffs = pywt.wavedec(signal[lead], wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])  # Remove approximation coefficients
        reconstructed = pywt.waverec(coeffs, wavelet)
        cleaned[lead] = reconstructed[:length]
    
    return cleaned


def bandpass_filter(signal, sample_rate, low_freq=1.0, high_freq=47.0, order=4):
    """Apply bandpass Butterworth filter (zero-phase, deterministic)"""
    nyquist = 0.5 * sample_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    filtered = sosfiltfilt(sos, signal, axis=-1)
    
    return filtered


def normalize_signal(signal, method='zscore'):
    """Normalize signal amplitude (deterministic)"""
    x = signal.copy()
    
    if method == 'zscore':
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
        std[(std == 0) | np.isnan(std)] = 1.0
        x = (x - mean) / std
    
    elif method == 'minmax':
        min_val = np.nanmin(x, axis=1, keepdims=True)
        max_val = np.nanmax(x, axis=1, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        x = 2.0 * (x - min_val) / range_val - 1.0
    
    elif method == 'robust':
        median = np.nanmedian(x, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(x - median), axis=1, keepdims=True)
        mad[mad == 0] = 1.0
        x = (x - median) / (1.4826 * mad)
    
    return x


def standardize_length(signal, target_length=4096):
    """Standardize signal length via cropping or padding (deterministic - center crop)"""
    leads, current_length = signal.shape
    
    if current_length < target_length:
        # Pad with edge values (not zeros for test data)
        pad_width = target_length - current_length
        signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='edge')
    
    elif current_length > target_length:
        # Center cropping for test/validation (NOT random)
        start = (current_length - target_length) // 2
        signal = signal[:, start:start + target_length]
    
    return signal


def preprocess_ecg_test(signal, header, config):
    """
    Complete ECG preprocessing pipeline for TEST data.
    NO augmentation, NO random operations - fully deterministic.
    """
    # Handle NaN values
    signal = np.nan_to_num(signal)
    
    # Transpose to (leads, samples)
    signal = signal.T
    
    # Get sampling frequency from header
    fs = get_sampling_frequency(header)
    if fs is None:
        fs = 500  # Default
    
    # 1. Resample to target frequency
    signal = resample_signal(signal, fs, config.TARGET_SAMPLING_RATE)
    
    # 2. Reflection padding for filter edge effects
    pad_width = int(config.EDGE_PAD_SECONDS * config.TARGET_SAMPLING_RATE)
    signal = np.pad(signal, ((0, 0), (pad_width, pad_width)), mode='reflect')
    
    # 3. Remove powerline interference (both 50Hz and 60Hz)
    signal = remove_powerline_interference(
        signal, 
        config.TARGET_SAMPLING_RATE, 
        config.POWERLINE_FREQS,
        config.NOTCH_Q_FACTOR
    )
    
    # 4. Remove baseline wander (wavelet)
    signal = remove_baseline_wavelet(signal, config.WAVELET_TYPE, config.WAVELET_LEVEL)
    
    # 5. Bandpass filter
    signal = bandpass_filter(
        signal, 
        config.TARGET_SAMPLING_RATE,
        config.BANDPASS_LOW,
        config.BANDPASS_HIGH
    )
    
    # 6. Remove padding
    signal = signal[:, pad_width:-pad_width]
    
    # 7. Trim edge artifacts
    signal[:, :config.EDGE_TRIM_SAMPLES] = 0.0
    signal[:, -config.EDGE_TRIM_SAMPLES:] = 0.0
    
    # 8. Standardize length (center crop - deterministic)
    signal = standardize_length(signal, config.ECG_LENGTH)
    
    # 9. Normalize amplitude (z-score - deterministic)
    signal = normalize_signal(signal, config.NORMALIZATION_METHOD)
    
    # NO AUGMENTATION for test data!
    
    return signal.astype(np.float32)


# ============================================================================
# HELPER FUNCTIONS (from helper_code.py)
# ============================================================================

def get_sampling_frequency(header):
    """Extract sampling frequency from header"""
    for line in header.split('\n'):
        if line.startswith('# Sampling Frequency:'):
            try:
                return int(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif line.startswith('# Frequency:'):
            try:
                return int(line.split(':')[1].strip().split()[0])
            except:
                pass
    return None


def load_signals(record_path):
    """Load ECG signals from .hea and .dat files"""
    # Remove .hea extension if present
    if record_path.endswith('.hea'):
        record_path = record_path[:-4]
    
    # Load header
    header = load_header(record_path + '.hea')
    
    # Try to load using wfdb
    try:
        import wfdb
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal.T  # (leads, samples)
        return signal, header
    except:
        # Fallback: load .dat file directly
        dat_path = record_path + '.dat'
        signal = load_dat_file(dat_path, header)
        return signal, header


def load_header(record_path):
    """Load header file"""
    if not record_path.endswith('.hea'):
        record_path = record_path + '.hea'
    
    with open(record_path, 'r') as f:
        header = f.read()
    
    return header


def load_dat_file(dat_path, header):
    """Load .dat file directly"""
    # Parse header to get number of leads and samples
    num_leads = 12
    
    # Read binary data
    with open(dat_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int16)
    
    # Try to infer shape from header
    try:
        for line in header.split('\n'):
            if not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    num_samples = int(parts[1])
                    break
    except:
        num_samples = len(data) // num_leads
    
    # Reshape to (leads, samples)
    signal = data[:num_leads * num_samples].reshape(num_leads, -1)
    
    return signal


def load_label(record_path):
    """Load label from header or CSV"""
    header = load_header(record_path)
    # Look for diagnosis in header
    for line in header.split('\n'):
        if line.startswith('# Diagnosis:'):
            diagnosis = line.split(':')[1].strip()
            if 'Chagas' in diagnosis or 'chagas' in diagnosis:
                return 1
    return 0


# ============================================================================
# CSV-BASED PREPROCESSING
# ============================================================================

def load_csv_metadata(csv_path):
    """Load metadata from CSV file"""
    df = pd.read_csv(csv_path)
    return df


def preprocess_record_from_csv(row, config):
    """
    Preprocess a single record using CSV metadata.
    
    Args:
        row: pandas Series with record information
        config: PreprocessingConfig object
    
    Returns:
        processed_signal, metadata_dict
    """
    record_id = row['record_id']
    dataset = row['dataset']
    chagas_label = row['chagas_label']
    file_path = row['file_path']
    
    # ========================================================================
    # CONCATENATE BASE PATH WITH CSV PATH
    # ========================================================================
    # Remove leading slash if present to avoid double slashes
    if file_path.startswith('/'):
        file_path = file_path[1:]
    
    # Concatenate base path with CSV path
    full_file_path = os.path.join(config.BASE_DATA_PATH, file_path)
    
    # Verify file exists
    if not os.path.exists(full_file_path):
        print(f"WARNING: File not found at {full_file_path}")
        print(f"  CSV path: {row['file_path']}")
        print(f"  Full path: {full_file_path}")
        return None, None
    # ========================================================================
    
    # Load signal and header
    try:
        signal, header = load_signals(full_file_path)
    except Exception as e:
        print(f"Error loading {record_id}: {e}")
        return None, None
    
    # Preprocess
    processed_signal = preprocess_ecg_test(signal, header, config)
    
    # Create metadata
    metadata = {
        'record_id': record_id,
        'dataset': dataset,
        'age': row.get('age', 'Unknown'),
        'sex': row.get('sex', 'Unknown'),
        'chagas_label': int(chagas_label),
        'original_fs': get_sampling_frequency(header),
        'target_fs': config.TARGET_SAMPLING_RATE,
        'original_shape': signal.shape,
        'processed_shape': processed_signal.shape,
        'file_path': file_path,
        'full_file_path': full_file_path
    }
    
    return processed_signal, metadata


def save_preprocessed_signal(signal, metadata, output_path, config):
    """Save preprocessed signal to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if config.SAVE_FORMAT == 'numpy':
        np.savez_compressed(
            output_path,
            signal=signal,
            metadata=metadata
        )
    elif config.SAVE_FORMAT == 'torch':
        torch.save({
            'signal': torch.from_numpy(signal),
            'metadata': metadata
        }, output_path)


def load_preprocessed_signal(input_path, config):
    """Load preprocessed signal from disk"""
    if config.SAVE_FORMAT == 'numpy':
        data = np.load(input_path, allow_pickle=True)
        signal = data['signal']
        metadata = data['metadata'].item() if 'metadata' in data else {}
    elif config.SAVE_FORMAT == 'torch':
        data = torch.load(input_path, map_location='cpu')
        signal = data['signal'].numpy()
        metadata = data['metadata']
    
    return signal, metadata


def preprocess_csv_dataset(csv_path, output_folder, config, verbose=True):
    """
    Preprocess dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file with record metadata
        output_folder: Path to save preprocessed data
        config: PreprocessingConfig object
        verbose: Print progress
    
    Returns:
        Dictionary with preprocessing statistics
    """
    start_time = datetime.now()
    
    if verbose:
        print("=" * 80)
        print("ECG PREPROCESSING PIPELINE - PHYSIONET CHALLENGE 2025")
        print("=" * 80)
        print(f"Starting preprocessing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"CSV file: {csv_path}")
        print(f"Output folder: {output_folder}")
        print(f"BASE_DATA_PATH: {config.BASE_DATA_PATH}")
        print(f"Test records limit: {config.TEST_RECORDS}")
        print(f"Target sampling rate: {config.TARGET_SAMPLING_RATE} Hz")
        print(f"Signal length: {config.ECG_LENGTH} samples")
        print("=" * 80)
    
    # Verify base path exists
    if not os.path.exists(config.BASE_DATA_PATH):
        print(f"ERROR: BASE_DATA_PATH does not exist: {config.BASE_DATA_PATH}")
        sys.exit(1)
    
    # Load CSV metadata
    df = load_csv_metadata(csv_path)
    
    if verbose:
        print(f"Total records in CSV: {len(df)}")
        print(f"Processing only {config.TEST_RECORDS} records for validation")
        print("-" * 80)
    
    # Limit to test records
    df_test = df.head(config.TEST_RECORDS)
    
    # Track statistics
    stats = {
        'total': 0,
        'success': 0,
        'errors': 0,
        'by_dataset': {},
        'by_label': {0: 0, 1: 0},
        'records_processed': [],
        'records_failed': []
    }
    
    # Process each record
    for idx, row in df_test.iterrows():
        try:
            record_id = row['record_id']
            
            if verbose:
                print(f"[{idx + 1}/{len(df_test)}] Processing: {record_id}")
            
            # Preprocess
            signal, metadata = preprocess_record_from_csv(row, config)
            
            if signal is None:
                stats['errors'] += 1
                stats['records_failed'].append(record_id)
                if verbose:
                    print(f"  ❌ Failed to process {record_id}")
                continue
            
            # Determine output path
            output_path = os.path.join(
                output_folder,
                f"{record_id}_processed.npz"
            )
            
            # Save preprocessed signal
            save_preprocessed_signal(signal, metadata, output_path, config)
            
            # Update statistics
            stats['total'] += 1
            stats['success'] += 1
            stats['records_processed'].append(record_id)
            
            dataset = row['dataset']
            if dataset not in stats['by_dataset']:
                stats['by_dataset'][dataset] = 0
            stats['by_dataset'][dataset] += 1
            
            label = int(row['chagas_label'])
            stats['by_label'][label] += 1
            
            if verbose:
                print(f"  ✅ Success - Shape: {signal.shape}, Label: {label}")
            
        except Exception as e:
            stats['errors'] += 1
            stats['records_failed'].append(row['record_id'])
            if verbose:
                print(f"  ❌ Error processing {row['record_id']}: {str(e)}")
            continue
    
    # Save statistics
    stats['duration_seconds'] = (datetime.now() - start_time).total_seconds()
    stats['config'] = {
        'target_fs': config.TARGET_SAMPLING_RATE,
        'ecg_length': config.ECG_LENGTH,
        'bandpass': [config.BANDPASS_LOW, config.BANDPASS_HIGH],
        'normalization': config.NORMALIZATION_METHOD,
        'base_data_path': config.BASE_DATA_PATH
    }
    
    stats_file = os.path.join(output_folder, 'preprocessing_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Save processed record list for train/test/val split
    records_file = os.path.join(output_folder, 'processed_records.txt')
    with open(records_file, 'w') as f:
        for record_id in stats['records_processed']:
            f.write(f"{record_id}\n")
    
    # Print summary
    if verbose:
        print("=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total records attempted: {stats['total']}")
        print(f"Successfully processed: {stats['success']}")
        print(f"Errors: {stats['errors']}")
        print(f"By dataset: {stats['by_dataset']}")
        print(f"By label: {stats['by_label']}")
        print(f"Duration: {stats['duration_seconds']:.1f} seconds")
        print(f"Output folder: {output_folder}")
        print("=" * 80)
        print("\n📁 Files created:")
        print(f"  - Preprocessed signals: {output_folder}/*.npz")
        print(f"  - Statistics: {stats_file}")
        print(f"  - Record list: {records_file}")
        print("=" * 80)
        print("\n⚠️  IMPORTANT: This preprocessing is TEST-SAFE (no augmentation)")
        print("   You can now split into train/test/val without data contamination!")
        print("=" * 80)
    
    return stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Paths (modify as needed)
    CSV_PATH = "/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/dataset_chagas_absolut.csv"  # Your CSV file
    OUTPUT_FOLDER = "./data/preprocessed"
    
    # Create configuration
    config = PreprocessingConfig()
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Run preprocessing
    stats = preprocess_csv_dataset(
        csv_path=CSV_PATH,
        output_folder=OUTPUT_FOLDER,
        config=config,
        verbose=True
    )
    
    print("\n✅ Preprocessing pipeline completed successfully!")
    print(f"📁 Preprocessed files saved to: {OUTPUT_FOLDER}")
    print(f"📊 Statistics saved to: {os.path.join(OUTPUT_FOLDER, 'preprocessing_stats.json')}")
    print(f"📝 Record list saved to: {os.path.join(OUTPUT_FOLDER, 'processed_records.txt')}")