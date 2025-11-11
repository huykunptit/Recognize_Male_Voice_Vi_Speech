#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script gộp tất cả file part trong male_only thành một file duy nhất
"""

import os
import pandas as pd
import glob

def merge_male_only_files():
    """Gộp tất cả file part trong male_only"""
    print("Gop tat ca file part trong male_only...")
    
    # Tìm tất cả file CSV trong male_only
    pattern = "super_metadata/male_only/ket_qua_cuoi_part_*.csv"
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print("ERROR - Khong tim thay file part nao!")
        return False
    
    print(f"Tim thay {len(csv_files)} file part")
    
    # Sắp xếp theo tên file
    csv_files.sort()
    
    # Gộp tất cả file
    all_dataframes = []
    
    for i, file_path in enumerate(csv_files):
        print(f"Dang doc file {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            all_dataframes.append(df)
            print(f"  - {len(df)} records")
        except Exception as e:
            print(f"  ERROR - Loi khi doc file: {e}")
            continue
    
    if not all_dataframes:
        print("ERROR - Khong doc duoc file nao!")
        return False
    
    # Gộp tất cả DataFrame
    print("\nDang gop tat ca DataFrame...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Tong so records: {len(merged_df)}")
    print(f"Columns: {list(merged_df.columns)}")
    
    # Lưu file gộp
    output_file = "super_metadata/male_only_merged.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nSUCCESS - Da luu file gop: {output_file}")
    print(f"Kich thuoc: {os.path.getsize(output_file):,} bytes")
    
    # Kiểm tra dữ liệu
    print("\nKiem tra du lieu:")
    feature_columns = [
        'pitch_mean', 'mfcc_1_mean', 'tempo', 'duration',
        'spectral_centroid_mean', 'rms_mean', 'loudness'
    ]
    
    for col in feature_columns:
        if col in merged_df.columns:
            non_zero = (merged_df[col] != 0).sum()
            total = len(merged_df)
            percentage = (non_zero / total) * 100
            print(f"  {col}: {non_zero}/{total} ({percentage:.1f}%) khac 0")
    
    return True

if __name__ == "__main__":
    merge_male_only_files()
