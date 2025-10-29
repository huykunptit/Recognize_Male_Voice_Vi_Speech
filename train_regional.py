#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script training với lọc theo vùng miền
"""

import os
import sys
import pandas as pd
import numpy as np
import librosa
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class RegionalAudioTrainer:
    def __init__(self):
        self.trainset_dir = "trainset"
        self.metadata_file = "metadata/trainset.csv"
        self.output_dir = "super_metadata/regional"
        
        # Tạo thư mục output
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_regional_distribution(self):
        """Lấy phân bố theo vùng miền"""
        try:
            df = pd.read_csv(self.metadata_file)
            print("Phan bo theo vung mien:")
            print(df['dialect'].value_counts())
            return df
        except Exception as e:
            print(f"Loi khi doc metadata: {e}")
            return None
    
    def filter_by_region(self, df, regions=None):
        """Lọc dữ liệu theo vùng miền"""
        if regions is None:
            regions = ['North', 'Central', 'South']  # Tất cả vùng
        
        if isinstance(regions, str):
            regions = [regions]
        
        filtered_df = df[df['dialect'].isin(regions)]
        print(f"Da loc {len(filtered_df)} files tu {len(df)} files")
        print(f"Vung mien duoc chon: {regions}")
        
        return filtered_df
    
    def extract_audio_features(self, audio_path):
        """Trích xuất đặc trưng âm thanh"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path)
            
            # Cắt xuống 20s nếu cần
            if len(y) / sr > 20:
                y = y[:int(20 * sr)]
            
            features = {}
            
            # 1. Pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            features['pitch_mean'] = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0
            features['pitch_std'] = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0.0
            
            # 2. Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # 3. MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(5):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            
            # 5. RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = float(np.mean(rms))
            
            # 6. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # 7. Duration
            features['duration'] = float(len(y) / sr)
            
            # 8. Loudness
            features['loudness'] = float(20 * np.log10(np.mean(np.abs(y)) + 1e-10))
            
            # 9. Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # 10. Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            
            # 11. Harmonic-to-noise ratio
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            hnr = np.mean(20 * np.log10(np.abs(y_harmonic) / (np.abs(y_percussive) + 1e-10)))
            features['hnr'] = float(hnr) if not np.isnan(hnr) and not np.isinf(hnr) else 0.0
            
            return features
            
        except Exception as e:
            print(f"Loi khi trich xuat dac trung tu {audio_path}: {e}")
            return None
    
    def train_regional_data(self, regions=None):
        """Training dữ liệu theo vùng miền"""
        print("Bat dau training theo vung mien...")
        print("=" * 60)
        
        # Đọc metadata
        df = self.get_regional_distribution()
        if df is None:
            return False
        
        # Lọc theo vùng miền
        filtered_df = self.filter_by_region(df, regions)
        
        if len(filtered_df) == 0:
            print("Khong co du lieu nao sau khi loc!")
            return False
        
        # Tạo danh sách features
        feature_columns = [
            'pitch_mean', 'pitch_std', 'spectral_centroid_mean', 'spectral_centroid_std',
            'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean',
            'zcr_mean', 'rms_mean', 'tempo', 'loudness', 'duration',
            'spectral_bandwidth_mean', 'spectral_flatness_mean', 'hnr'
        ]
        
        # Khởi tạo DataFrame kết quả
        results = []
        
        # Xử lý từng file
        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Training"):
            audio_name = row['audio_name']
            audio_path = os.path.join(self.trainset_dir, audio_name)
            
            if not os.path.exists(audio_path):
                print(f"Khong tim thay file: {audio_path}")
                continue
            
            # Trích xuất đặc trưng
            features = self.extract_audio_features(audio_path)
            
            if features is None:
                # Tạo features mặc định
                features = {col: 0.0 for col in feature_columns}
            
            # Thêm thông tin metadata
            result_row = {
                'audio_name': audio_name,
                'speaker': row['speaker'],
                'dialect': row['dialect'],
                'gender': row['gender']
            }
            
            # Thêm features
            result_row.update(features)
            results.append(result_row)
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame(results)
        
        # Lưu kết quả
        region_name = "_".join(regions) if isinstance(regions, list) else regions
        output_file = os.path.join(self.output_dir, f"regional_{region_name.lower()}.csv")
        
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\nDa luu ket qua vao: {output_file}")
        print(f"Tong so files da xu ly: {len(result_df)}")
        
        # Thống kê theo vùng miền
        print("\nThong ke theo vung mien:")
        print(result_df['dialect'].value_counts())
        
        return True

def main():
    """Hàm chính"""
    print("ViSpeech - Regional Training")
    print("=" * 60)
    
    trainer = RegionalAudioTrainer()
    
    # Hiển thị menu lựa chọn
    print("Chon vung mien de training:")
    print("1. Tat ca vung mien (North, Central, South)")
    print("2. Chi mien Bac (North)")
    print("3. Chi mien Trung (Central)")
    print("4. Chi mien Nam (South)")
    print("5. Tu chon vung mien")
    
    choice = input("\nNhap lua chon (1-5): ").strip()
    
    regions = None
    
    if choice == "1":
        regions = ['North', 'Central', 'South']
    elif choice == "2":
        regions = ['North']
    elif choice == "3":
        regions = ['Central']
    elif choice == "4":
        regions = ['South']
    elif choice == "5":
        print("\nCac vung mien co san: North, Central, South")
        regions_input = input("Nhap vung mien (cach nhau boi dau phay): ").strip()
        regions = [r.strip() for r in regions_input.split(',')]
    else:
        print("Lua chon khong hop le!")
        return
    
    print(f"\nBat dau training cho vung mien: {regions}")
    
    # Thực hiện training
    success = trainer.train_regional_data(regions)
    
    if success:
        print("\n✅ Training hoan thanh!")
        print("📁 Ket qua duoc luu trong folder 'super_metadata/regional/'")
    else:
        print("\n❌ Training that bai!")

if __name__ == "__main__":
    main()
