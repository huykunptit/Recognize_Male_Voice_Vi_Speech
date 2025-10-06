#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tạo super metadata với 15 trường thông tin âm thanh chi tiết
"""

import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
from tqdm import tqdm
import warnings
import sys
warnings.filterwarnings('ignore')

# Set encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class SuperMetadataCreator:
    def __init__(self):
        self.metadata_folder = "metadata"
        self.super_metadata_folder = "super_metadata"
        self.audio_folders = {
            "trainset": "trainset",
            "clean_testset": "clean_testset", 
            "noisy_testset": "noisy_testset"
        }
        
        # Tạo folder super_metadata
        Path(self.super_metadata_folder).mkdir(exist_ok=True)
        print(f"Đã tạo folder: {self.super_metadata_folder}")
    
    def extract_advanced_audio_features(self, audio_path):
        """Trích xuất 15 đặc trưng âm thanh chi tiết"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path)
            
            features = {}
            
            # 1. Độ cao giọng (Pitch) - F0
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            features['pitch_mean'] = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
            features['pitch_std'] = np.nanstd(f0) if not np.all(np.isnan(f0)) else 0
            features['pitch_range'] = np.nanmax(f0) - np.nanmin(f0) if not np.all(np.isnan(f0)) else 0
            
            # 2. Độ trầm bổng (Spectral Centroid)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # 3. Độ rõ ràng (Spectral Rolloff)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # 4. Zero Crossing Rate (Tần suất đổi dấu)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 5. MFCC (Mel-frequency cepstral coefficients) - 13 hệ số
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
            # 6. Chroma features (Đặc trưng âm sắc)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # 7. Spectral contrast (Độ tương phản phổ)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = np.mean(contrast)
            features['spectral_contrast_std'] = np.std(contrast)
            
            # 8. Tonnetz (Đặc trưng hòa âm)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz)
            features['tonnetz_std'] = np.std(tonnetz)
            
            # 9. RMS Energy (Năng lượng âm thanh)
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            features['rms_max'] = np.max(rms)
            features['rms_min'] = np.min(rms)
            
            # 10. Tempo (Nhịp độ)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # 11. Duration (Thời lượng)
            features['duration'] = len(y) / sr
            
            # 12. Loudness (Độ to) - dB
            features['loudness'] = 20 * np.log10(np.mean(np.abs(y)) + 1e-10)
            features['loudness_peak'] = 20 * np.log10(np.max(np.abs(y)) + 1e-10)
            
            # 13. Spectral bandwidth (Băng thông phổ)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # 14. Spectral flatness (Độ phẳng phổ)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            features['spectral_flatness_std'] = np.std(spectral_flatness)
            
            # 15. Harmonic-to-noise ratio (Tỷ lệ hài hòa/nhiễu)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            hnr = np.mean(20 * np.log10(np.abs(y_harmonic) / (np.abs(y_percussive) + 1e-10)))
            features['hnr'] = hnr if not np.isnan(hnr) and not np.isinf(hnr) else 0
            
            # Thêm một số đặc trưng bổ sung
            # 16. Spectral slope (Độ dốc phổ)
            spectral_slope = librosa.feature.spectral_slope(y=y)[0]
            features['spectral_slope_mean'] = np.mean(spectral_slope)
            features['spectral_slope_std'] = np.std(spectral_slope)
            
            # 17. Spectral kurtosis (Độ nhọn phổ)
            spectral_kurtosis = librosa.feature.spectral_kurtosis(y=y)[0]
            features['spectral_kurtosis_mean'] = np.mean(spectral_kurtosis)
            features['spectral_kurtosis_std'] = np.std(spectral_kurtosis)
            
            # 18. Spectral skewness (Độ lệch phổ)
            spectral_skewness = librosa.feature.spectral_skewness(y=y)[0]
            features['spectral_skewness_mean'] = np.mean(spectral_skewness)
            features['spectral_skewness_std'] = np.std(spectral_skewness)
            
            # 19. Onset strength (Độ mạnh bắt đầu)
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength_mean'] = np.mean(onset_strength)
            features['onset_strength_std'] = np.std(onset_strength)
            
            # 20. Spectral flux (Dòng phổ)
            spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
            features['spectral_flux'] = spectral_flux
            
            return features
            
        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng từ {audio_path}: {e}")
            return None
    
    def process_metadata_file(self, csv_file, audio_folder):
        """Xử lý một file metadata và tạo super metadata"""
        try:
            print(f"\n=== Xử lý {csv_file} ===")
            
            # Đọc file metadata gốc với encoding
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='latin-1')
            
            print(f"Đã đọc {len(df)} records từ {csv_file}")
            
            # Tạo danh sách để lưu kết quả
            super_data = []
            
            # Xử lý từng file audio
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Trích xuất đặc trưng"):
                try:
                    audio_name = row['audio_name']
                    audio_path = os.path.join(audio_folder, audio_name)
                    
                    # Tạo record mới với thông tin gốc
                    new_record = {
                        'audio_name': audio_name,
                        'dialect': row['dialect'],
                        'gender': row['gender'],
                        'speaker': row['speaker']
                    }
                    
                    # Trích xuất đặc trưng âm thanh
                    if os.path.exists(audio_path):
                        features = self.extract_advanced_audio_features(audio_path)
                        if features:
                            new_record.update(features)
                        else:
                            # Nếu không trích xuất được, điền giá trị mặc định
                            default_features = self.get_default_features()
                            new_record.update(default_features)
                    else:
                        print(f"Không tìm thấy file: {audio_path}")
                        default_features = self.get_default_features()
                        new_record.update(default_features)
                    
                    super_data.append(new_record)
                    
                except Exception as e:
                    print(f"Lỗi khi xử lý file {audio_name}: {e}")
                    # Thêm record với giá trị mặc định
                    default_features = self.get_default_features()
                    new_record = {
                        'audio_name': row['audio_name'],
                        'dialect': row['dialect'],
                        'gender': row['gender'],
                        'speaker': row['speaker']
                    }
                    new_record.update(default_features)
                    super_data.append(new_record)
            
            # Tạo DataFrame và lưu
            super_df = pd.DataFrame(super_data)
            
            # Tạo tên file output
            output_file = os.path.join(self.super_metadata_folder, os.path.basename(csv_file))
            super_df.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"Đã tạo super metadata: {output_file}")
            print(f"Số cột: {len(super_df.columns)}")
            print(f"Số dòng: {len(super_df)}")
            
            return super_df
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {csv_file}: {e}")
            return None
    
    def get_default_features(self):
        """Trả về các đặc trưng mặc định khi không trích xuất được"""
        return {
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
    
    def create_all_super_metadata(self):
        """Tạo tất cả super metadata files"""
        print("=== Tạo Super Metadata với 15+ trường thông tin âm thanh ===\n")
        
        # Xử lý từng file metadata
        for csv_name, audio_folder in self.audio_folders.items():
            csv_path = os.path.join(self.metadata_folder, f"{csv_name}.csv")
            
            if os.path.exists(csv_path):
                self.process_metadata_file(csv_path, audio_folder)
            else:
                print(f"Không tìm thấy file: {csv_path}")
        
        print(f"\n=== Hoàn thành! ===")
        print(f"Super metadata đã được tạo trong folder: {self.super_metadata_folder}")
        print("Các file bao gồm:")
        for file in os.listdir(self.super_metadata_folder):
            if file.endswith('.csv'):
                print(f"  - {file}")

def main():
    """Hàm chính"""
    creator = SuperMetadataCreator()
    creator.create_all_super_metadata()

if __name__ == "__main__":
    main()
