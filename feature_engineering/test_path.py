#!/usr/bin/env python
"""
Test script to verify file paths before preprocessing
"""

import pandas as pd
import os

CSV_PATH = "/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/dataset_chagas_absolut.csv"
BASE_PATH = "/media/mpsilva-lx/Dados/Recursos"

print("=" * 80)
print("PATH VERIFICATION TEST")
print("=" * 80)
print(f"CSV_PATH: {CSV_PATH}")
print(f"BASE_PATH: {BASE_PATH}")
print("=" * 80)

# Check if CSV exists
if not os.path.exists(CSV_PATH):
    print(f"❌ ERROR: CSV file not found: {CSV_PATH}")
    exit(1)

# Check if base path exists
if not os.path.exists(BASE_PATH):
    print(f"❌ ERROR: BASE_PATH does not exist: {BASE_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)

print(f"\nTotal records in CSV: {len(df)}")
print(f"\nTesting paths for first {min(5, len(df))} records:")
print("-" * 80)

success_count = 0
error_count = 0

for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    csv_path = row['file_path']
    
    if csv_path.startswith('/'):
        csv_path = csv_path[1:]
    
    full_path = os.path.join(BASE_PATH, csv_path)
    exists = os.path.exists(full_path)
    
    if exists:
        success_count += 1
        status = "✅ YES"
    else:
        error_count += 1
        status = "❌ NO"
    
    print(f"Record {row['record_id']}:")
    print(f"  CSV path: {row['file_path']}")
    print(f"  Full path: {full_path}")
    print(f"  Exists: {status}")
    print("-" * 80)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Successful: {success_count}/{min(5, len(df))}")
print(f"Errors: {error_count}/{min(5, len(df))}")
print("=" * 80)

if error_count > 0:
    print("\n⚠️  WARNING: Some files not found! Check BASE_PATH configuration.")
    print("   Modify BASE_DATA_PATH in preprocess_ecg_csv.py")
else:
    print("\n✅ All paths verified successfully!")
    print("   You can now run: python preprocess_ecg_csv.py")
print("=" * 80)