#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script test tạo super metadata đơn giản
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_simple_super_metadata():
    """Tạo super metadata đơn giản với các trường mặc định"""
    print("=== Tạo Super Metadata Đơn giản ===\n")
    
    # Tạo folder super_metadata
    super_metadata_folder = "super_metadata"
    Path(super_metadata_folder).mkdir(exist_ok=True)
    print(f"Đã tạo folder: {super_metadata_folder}")
    
    # Các file metadata cần xử lý
    metadata_files = [
        ("metadata/clean_testset.csv", "clean_testset"),
        ("metadata/noisy_testset.csv", "noisy_testset"), 
        ("metadata/trainset.csv", "trainset")
    ]
    
    for csv_file, dataset_name in metadata_files:
        if os.path.exists(csv_file):
            print(f"\n=== Xử lý {csv_file} ===")
            
            try:
                # Đọc file gốc
                df = pd.read_csv(csv_file, encoding='utf-8')
                print(f"Đã đọc {len(df)} records từ {csv_file}")
                
                # Tạo các trường mặc định cho audio features
                default_features = {
                    'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0,
                    'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
                    'spectral_rolloff_mean': 0, 'spectral_rolloff_std': 0,
                    'zcr_mean': 0, 'zcr_std': 0,
                    'mfcc_1_mean': 0, 'mfcc_1_std': 0, 'mfcc_2_mean': 0, 'mfcc_2_std': 0,
                    'mfcc_3_mean': 0, 'mfcc_3_std': 0, 'mfcc_4_mean': 0, 'mfcc_4_std': 0,
                    'mfcc_5_mean': 0, 'mfcc_5_std': 0, 'mfcc_6_mean': 0, 'mfcc_6_std': 0,
                    'mfcc_7_mean': 0, 'mfcc_7_std': 0, 'mfcc_8_mean': 0, 'mfcc_8_std': 0,
                    'mfcc_9_mean': 0, 'mfcc_9_std': 0, 'mfcc_10_mean': 0, 'mfcc_10_std': 0,
                    'mfcc_11_mean': 0, 'mfcc_11_std': 0, 'mfcc_12_mean': 0, 'mfcc_12_std': 0,
                    'mfcc_13_mean': 0, 'mfcc_13_std': 0,
                    'chroma_mean': 0, 'chroma_std': 0,
                    'spectral_contrast_mean': 0, 'spectral_contrast_std': 0,
                    'tonnetz_mean': 0, 'tonnetz_std': 0,
                    'rms_mean': 0, 'rms_std': 0, 'rms_max': 0, 'rms_min': 0,
                    'tempo': 0, 'duration': 0, 'loudness': 0, 'loudness_peak': 0,
                    'spectral_bandwidth_mean': 0, 'spectral_bandwidth_std': 0,
                    'spectral_flatness_mean': 0, 'spectral_flatness_std': 0,
                    'hnr': 0, 'spectral_slope_mean': 0, 'spectral_slope_std': 0,
                    'spectral_kurtosis_mean': 0, 'spectral_kurtosis_std': 0,
                    'spectral_skewness_mean': 0, 'spectral_skewness_std': 0,
                    'onset_strength_mean': 0, 'onset_strength_std': 0,
                    'spectral_flux': 0
                }
                
                # Thêm các trường mặc định vào DataFrame
                for feature, value in default_features.items():
                    df[feature] = value
                
                # Lưu file super metadata
                output_file = os.path.join(super_metadata_folder, f"{dataset_name}.csv")
                df.to_csv(output_file, index=False, encoding='utf-8')
                
                print(f"Đã tạo super metadata: {output_file}")
                print(f"Số cột: {len(df.columns)}")
                print(f"Số dòng: {len(df)}")
                
            except Exception as e:
                print(f"Lỗi khi xử lý {csv_file}: {e}")
        else:
            print(f"Không tìm thấy file: {csv_file}")
    
    print(f"\n=== Hoàn thành! ===")
    print(f"Super metadata đã được tạo trong folder: {super_metadata_folder}")

if __name__ == "__main__":
    create_simple_super_metadata()
