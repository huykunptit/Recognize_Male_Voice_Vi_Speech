#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ứng dụng tìm kiếm giọng nói tương tự - Giao diện Windows GUI (Enhanced Version)
"""

import os
import sys
import json
import warnings
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd

# Xử lý lỗi numba/librosa trên Windows
LIBROSA_AVAILABLE = False
try:
    import librosa
    LIBROSA_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"⚠️  Warning: librosa không thể import: {e}")

import soundfile as sf
from scipy.stats import kurtosis, skew

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib

# Audio playback and recording
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False
    print("Warning: pygame không khả dụng, không thể phát audio")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice không khả dụng, không thể ghi âm")

warnings.filterwarnings('ignore')

# Set encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass

# Config
TRAINED_MODEL_FILE = 'trained_model.json'
SCALER_FILE = 'scaler.joblib'
KNN_MODEL_FILE = 'knn_model.joblib'
SUPER_METADATA_FOLDER = "super_metadata/male_only"
SPEAKER_DB_FILE = "speaker_database.csv"

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class VoiceSearchEngine:
    def __init__(self):
        self.super_metadata_folder = SUPER_METADATA_FOLDER
        self.speaker_db_file = SPEAKER_DB_FILE
        self.scaler = None
        self.knn_model = None
        self.feature_columns = None
        self.df_train = None
        self.speaker_db = None
        self.input_dialect = None  # Dialect của input audio
        
    def load_speaker_database(self):
        """Load speaker database với tên tiếng Việt"""
        try:
            self.speaker_db = pd.read_csv(self.speaker_db_file, encoding='utf-8')
            return True
        except Exception as e:
            print(f"Lỗi khi load speaker database: {e}")
            return False
    
    def load_training_data(self):
        """Load và merge tất cả CSV files từ super_metadata/male_only"""
        all_data = []
        csv_files = sorted(Path(self.super_metadata_folder).glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                all_data.append(df)
            except Exception as e:
                print(f"Lỗi khi load {csv_file.name}: {e}")
        
        if not all_data:
            raise ValueError("Không tìm thấy dữ liệu training!")
        
        self.df_train = pd.concat(all_data, ignore_index=True)
        return True
    
    def get_feature_columns(self):
        """Lấy danh sách các cột features"""
        if self.feature_columns is None:
            exclude_cols = ['audio_name', 'dialect', 'gender', 'speaker']
            self.feature_columns = [col for col in self.df_train.columns 
                                   if col not in exclude_cols]
        return self.feature_columns
    
    def train_model(self, progress_callback=None):
        """Train KNN model từ dữ liệu training"""
        if progress_callback:
            progress_callback("Đang load dữ liệu training...")
        
        self.load_training_data()
        self.load_speaker_database()
        
        feature_cols = self.get_feature_columns()
        
        if progress_callback:
            progress_callback("Đang chuẩn hóa dữ liệu...")
        
        X_train = self.df_train[feature_cols].fillna(0).values
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if progress_callback:
            progress_callback("Đang train KNN model...")
        
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
        self.knn_model.fit(X_train_scaled)
        
        # Lưu model
        joblib.dump(self.scaler, SCALER_FILE)
        joblib.dump(self.knn_model, KNN_MODEL_FILE)
        
        training_info = {
            'trained_at': datetime.now().isoformat(),
            'num_samples': len(self.df_train),
            'num_features': len(feature_cols),
            'feature_columns': feature_cols,
            'model_type': 'KNN',
            'k_neighbors': 10,
            'metric': 'cosine',
            'training_files': [f.name for f in Path(self.super_metadata_folder).glob("*.csv")]
        }
        
        with open(TRAINED_MODEL_FILE, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        if progress_callback:
            progress_callback("Hoàn thành!")
        
        return training_info
    
    def load_trained_model(self):
        """Load model đã train từ file"""
        try:
            with open(TRAINED_MODEL_FILE, 'r', encoding='utf-8') as f:
                training_info = json.load(f)
            
            self.scaler = joblib.load(SCALER_FILE)
            self.knn_model = joblib.load(KNN_MODEL_FILE)
            self.load_training_data()
            self.load_speaker_database()
            self.feature_columns = training_info['feature_columns']
            
            return training_info
        except Exception as e:
            print(f"Không tìm thấy model đã train: {e}")
            return None
    
    def compute_spectral_slope(self, y, sr, n_fft=2048, hop_length=512):
        """
        Compute spectral slope per frame by fitting a line to log-magnitude vs frequency.
        Returns array of slopes (one per frame).
        """
        try:
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        except Exception:
            return np.zeros(1, dtype=float)
        freqs = np.linspace(0, sr / 2, S.shape[0])
        slopes = []
        for i in range(S.shape[1]):
            mag = S[:, i]
            if np.all(mag == 0):
                slopes.append(0.0)
                continue
            mag_log = np.log1p(mag)
            try:
                a, _ = np.polyfit(freqs, mag_log, 1)
            except Exception:
                a = 0.0
            slopes.append(float(a))
        return np.array(slopes)
    
    def get_default_features(self):
        """Return default feature dict (zeros) - giống hệt training"""
        return {
            'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0,
            'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0, 'spectral_rolloff_std': 0.0,
            'zcr_mean': 0.0, 'zcr_std': 0.0,
            'mfcc_1_mean': 0.0, 'mfcc_1_std': 0.0, 'mfcc_2_mean': 0.0, 'mfcc_2_std': 0.0,
            'mfcc_3_mean': 0.0, 'mfcc_3_std': 0.0, 'mfcc_4_mean': 0.0, 'mfcc_4_std': 0.0,
            'mfcc_5_mean': 0.0, 'mfcc_5_std': 0.0, 'mfcc_6_mean': 0.0, 'mfcc_6_std': 0.0,
            'mfcc_7_mean': 0.0, 'mfcc_7_std': 0.0, 'mfcc_8_mean': 0.0, 'mfcc_8_std': 0.0,
            'mfcc_9_mean': 0.0, 'mfcc_9_std': 0.0, 'mfcc_10_mean': 0.0, 'mfcc_10_std': 0.0,
            'mfcc_11_mean': 0.0, 'mfcc_11_std': 0.0, 'mfcc_12_mean': 0.0, 'mfcc_12_std': 0.0,
            'mfcc_13_mean': 0.0, 'mfcc_13_std': 0.0,
            'chroma_mean': 0.0, 'chroma_std': 0.0,
            'spectral_contrast_mean': 0.0, 'spectral_contrast_std': 0.0,
            'tonnetz_mean': 0.0, 'tonnetz_std': 0.0,
            'rms_mean': 0.0, 'rms_std': 0.0, 'rms_max': 0.0, 'rms_min': 0.0,
            'tempo': 0.0, 'duration': 0.0, 'loudness': 0.0, 'loudness_peak': 0.0,
            'spectral_bandwidth_mean': 0.0, 'spectral_bandwidth_std': 0.0,
            'spectral_flatness_mean': 0.0, 'spectral_flatness_std': 0.0,
            'hnr': 0.0, 'spectral_slope_mean': 0.0, 'spectral_slope_std': 0.0,
            'spectral_kurtosis_mean': 0.0, 'spectral_kurtosis_std': 0.0,
            'spectral_skewness_mean': 0.0, 'spectral_skewness_std': 0.0,
            'onset_strength_mean': 0.0, 'onset_strength_std': 0.0,
            'spectral_flux': 0.0
        }
    
    def extract_audio_features(self, audio_path, progress_callback=None):
        """Extract audio features - GIỐNG HỆT train_audio_features.py"""
        global LIBROSA_AVAILABLE
        
        # Đảm bảo librosa được import
        if not LIBROSA_AVAILABLE:
            try:
                import librosa
                LIBROSA_AVAILABLE = True
            except ImportError as e:
                raise ImportError(f"Không thể import librosa: {e}. Vui lòng cài đặt: pip install librosa")
        
        import librosa
        defaults = self.get_default_features()
        
        try:
            if progress_callback:
                progress_callback("Đang load file audio...")
            
            # Kiểm tra file tồn tại
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"File không tồn tại: {audio_path}")
            
            # Load audio file - GIỐNG HỆT training (prefer soundfile)
            try:
                y, sr = sf.read(audio_path, always_2d=False)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
            except Exception:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            if y.size == 0:
                raise ValueError("Empty audio")
            
            # Convert to float và normalize - GIỐNG HỆT training
            y = librosa.util.normalize(y.astype(float))
            
            features = {}
            
            if progress_callback:
                progress_callback("Đang trích xuất features...")
            
            # Duration
            duration = float(len(y) / float(sr))
            features['duration'] = duration
            
            # RMS
            try:
                rms = librosa.feature.rms(y=y)[0]
                features['rms_mean'] = float(np.mean(rms))
                features['rms_std'] = float(np.std(rms))
                features['rms_max'] = float(np.max(rms))
                features['rms_min'] = float(np.min(rms))
            except Exception:
                features.update({k: defaults[k] for k in ['rms_mean','rms_std','rms_max','rms_min']})
            
            # Spectral Centroid
            try:
                sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                features['spectral_centroid_mean'] = float(np.mean(sc))
                features['spectral_centroid_std'] = float(np.std(sc))
            except Exception:
                features['spectral_centroid_mean'] = defaults['spectral_centroid_mean']
                features['spectral_centroid_std'] = defaults['spectral_centroid_std']
            
            # Spectral Rolloff
            try:
                roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                features['spectral_rolloff_mean'] = float(np.mean(roll))
                features['spectral_rolloff_std'] = float(np.std(roll))
            except Exception:
                features['spectral_rolloff_mean'] = defaults['spectral_rolloff_mean']
                features['spectral_rolloff_std'] = defaults['spectral_rolloff_std']
            
            # ZCR
            try:
                z = librosa.feature.zero_crossing_rate(y)[0]
                features['zcr_mean'] = float(np.mean(z))
                features['zcr_std'] = float(np.std(z))
            except Exception:
                features['zcr_mean'] = defaults['zcr_mean']
                features['zcr_std'] = defaults['zcr_std']
            
            # MFCCs (13)
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                for i in range(13):
                    features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
                    features[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
            except Exception:
                for i in range(13):
                    features[f'mfcc_{i+1}_mean'] = defaults[f'mfcc_{i+1}_mean']
                    features[f'mfcc_{i+1}_std'] = defaults[f'mfcc_{i+1}_std']
            
            # Chroma
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                features['chroma_mean'] = float(np.mean(chroma))
                features['chroma_std'] = float(np.std(chroma))
            except Exception:
                features['chroma_mean'] = defaults['chroma_mean']
                features['chroma_std'] = defaults['chroma_std']
            
            # Spectral Contrast
            try:
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features['spectral_contrast_mean'] = float(np.mean(contrast))
                features['spectral_contrast_std'] = float(np.std(contrast))
            except Exception:
                features['spectral_contrast_mean'] = defaults['spectral_contrast_mean']
                features['spectral_contrast_std'] = defaults['spectral_contrast_std']
            
            # Tonnetz
            try:
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
                features['tonnetz_mean'] = float(np.mean(tonnetz))
                features['tonnetz_std'] = float(np.std(tonnetz))
            except Exception:
                features['tonnetz_mean'] = defaults['tonnetz_mean']
                features['tonnetz_std'] = defaults['tonnetz_std']
            
            # Tempo
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = float(tempo)
            except Exception:
                features['tempo'] = defaults['tempo']
            
            # Onset Strength
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                features['onset_strength_mean'] = float(np.mean(onset_env))
                features['onset_strength_std'] = float(np.std(onset_env))
            except Exception:
                features['onset_strength_mean'] = defaults['onset_strength_mean']
                features['onset_strength_std'] = defaults['onset_strength_std']
            
            # Spectral Bandwidth & Flatness
            try:
                bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                features['spectral_bandwidth_mean'] = float(np.mean(bw))
                features['spectral_bandwidth_std'] = float(np.std(bw))
            except Exception:
                features['spectral_bandwidth_mean'] = defaults['spectral_bandwidth_mean']
                features['spectral_bandwidth_std'] = defaults['spectral_bandwidth_std']
            
            try:
                flat = librosa.feature.spectral_flatness(y=y)[0]
                features['spectral_flatness_mean'] = float(np.mean(flat))
                features['spectral_flatness_std'] = float(np.std(flat))
            except Exception:
                features['spectral_flatness_mean'] = defaults['spectral_flatness_mean']
                features['spectral_flatness_std'] = defaults['spectral_flatness_std']
            
            # Spectral Slope - Dùng hàm compute_spectral_slope
            try:
                slopes = self.compute_spectral_slope(y, sr)
                features['spectral_slope_mean'] = float(np.mean(slopes))
                features['spectral_slope_std'] = float(np.std(slopes))
            except Exception:
                features['spectral_slope_mean'] = defaults['spectral_slope_mean']
                features['spectral_slope_std'] = defaults['spectral_slope_std']
            
            # Spectral Kurtosis & Skewness - Từ STFT
            try:
                S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
                spec_kurt = kurtosis(S, axis=0, fisher=True, nan_policy='omit')
                spec_skew = skew(S, axis=0, nan_policy='omit')
                features['spectral_kurtosis_mean'] = float(np.nanmean(spec_kurt))
                features['spectral_kurtosis_std'] = float(np.nanstd(spec_kurt))
                features['spectral_skewness_mean'] = float(np.nanmean(spec_skew))
                features['spectral_skewness_std'] = float(np.nanstd(spec_skew))
            except Exception:
                features['spectral_kurtosis_mean'] = defaults['spectral_kurtosis_mean']
                features['spectral_kurtosis_std'] = defaults['spectral_kurtosis_std']
                features['spectral_skewness_mean'] = defaults['spectral_skewness_mean']
                features['spectral_skewness_std'] = defaults['spectral_skewness_std']
            
            # Spectral Flux - Tính từ STFT diff
            try:
                S = np.abs(librosa.stft(y))
                flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
                features['spectral_flux'] = float(np.mean(flux)) if flux.size > 0 else 0.0
            except Exception:
                features['spectral_flux'] = defaults['spectral_flux']
            
            # Pitch - try pyin, fallback to yin, fallback to zeros
            try:
                f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
                if f0 is None:
                    raise Exception("pyin returned None")
                f0_clean = f0[~np.isnan(f0)]
                if f0_clean.size > 0:
                    features['pitch_mean'] = float(np.mean(f0_clean))
                    features['pitch_std'] = float(np.std(f0_clean))
                    features['pitch_range'] = float(np.max(f0_clean) - np.min(f0_clean))
                else:
                    features['pitch_mean'] = defaults['pitch_mean']
                    features['pitch_std'] = defaults['pitch_std']
                    features['pitch_range'] = defaults['pitch_range']
            except Exception:
                try:
                    f0_yin = librosa.yin(y, fmin=50, fmax=500, sr=sr)
                    f0_clean = f0_yin[~np.isnan(f0_yin)]
                    if f0_clean.size > 0:
                        features['pitch_mean'] = float(np.mean(f0_clean))
                        features['pitch_std'] = float(np.std(f0_clean))
                        features['pitch_range'] = float(np.max(f0_clean) - np.min(f0_clean))
                    else:
                        features['pitch_mean'] = defaults['pitch_mean']
                        features['pitch_std'] = defaults['pitch_std']
                        features['pitch_range'] = defaults['pitch_range']
                except Exception:
                    features['pitch_mean'] = defaults['pitch_mean']
                    features['pitch_std'] = defaults['pitch_std']
                    features['pitch_range'] = defaults['pitch_range']
            
            # Loudness - Tính từ rms_mean (GIỐNG HỆT training)
            try:
                features['loudness'] = 20 * np.log10(features.get('rms_mean', 1e-6) + 1e-6)
                features['loudness_peak'] = 20 * np.log10(features.get('rms_max', 1e-6) + 1e-6)
            except Exception:
                features['loudness'] = defaults['loudness']
                features['loudness_peak'] = defaults['loudness_peak']
            
            # HNR - Giữ default (GIỐNG HỆT training)
            features['hnr'] = defaults['hnr']
            
            # Merge defaults for any missing keys
            for k, v in defaults.items():
                if k not in features:
                    features[k] = v
            
            return features
            
        except Exception as e:
            error_msg = f"Lỗi khi trích xuất features từ {audio_path}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            # Return defaults thay vì raise để không làm gián đoạn
            return defaults
    
    def detect_dialect_from_features(self, features):
        """Dự đoán dialect từ features (nếu có thể) - placeholder function"""
        # Có thể implement logic phức tạp hơn dựa trên features
        # Hiện tại chỉ return None để dùng filter manual
        return None
    
    def search_similar_voices(self, audio_path, k=10, progress_callback=None, 
                             filter_dialect=None, boost_same_dialect=True):
        """Tìm K giọng nói tương tự nhất
        
        Args:
            audio_path: Đường dẫn file audio
            k: Số kết quả cần tìm
            progress_callback: Callback function
            filter_dialect: Filter theo dialect cụ thể (None = không filter)
            boost_same_dialect: Nếu True, boost similarity cho cùng dialect
        """
        try:
            # Kiểm tra model đã được load đầy đủ chưa
            if self.knn_model is None:
                raise ValueError("KNN model chưa được load. Vui lòng train hoặc load model trước!")
            
            if self.scaler is None:
                raise ValueError("Scaler chưa được load. Vui lòng train hoặc load model trước!")
            
            if self.df_train is None:
                raise ValueError("Training data chưa được load. Vui lòng train hoặc load model trước!")
            
            if self.feature_columns is None or len(self.feature_columns) == 0:
                raise ValueError("Feature columns chưa được định nghĩa. Vui lòng train model lại!")
            
            # Trích xuất features
            if progress_callback:
                progress_callback("Đang trích xuất features từ file audio...")
            
            features = self.extract_audio_features(audio_path, progress_callback)
            if features is None:
                raise ValueError("Không thể trích xuất features từ file audio")
            
            # Chuẩn bị feature vector - Đảm bảo tất cả features có giá trị
            feature_cols = self.get_feature_columns()
            
            # Kiểm tra và fill missing features với defaults
            defaults = self.get_default_features()
            for col in feature_cols:
                if col not in features:
                    features[col] = defaults.get(col, 0.0)
            
            # Tạo feature vector theo đúng thứ tự
            feature_vector = np.array([features.get(col, 0.0) for col in feature_cols]).reshape(1, -1)
            
            # Kiểm tra NaN hoặc Inf
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                print("Warning: Có NaN hoặc Inf trong features, thay thế bằng 0")
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Chuẩn hóa
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            if progress_callback:
                progress_callback("Đang tìm kiếm giọng nói tương tự...")
            
            # Tìm nhiều neighbors hơn nếu cần filter
            search_k = k * 3 if filter_dialect else k
            
            # Tìm K nearest neighbors
            distances, indices = self.knn_model.kneighbors(feature_vector_scaled, n_neighbors=min(search_k, len(self.df_train)))
            
            # Lấy thông tin các samples tương tự
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                sample = self.df_train.iloc[idx]
                base_similarity = (1 - dist) * 100
                
                sample_dialect = sample.get('dialect', 'N/A')
                
                # Boost similarity nếu cùng dialect
                similarity = base_similarity
                if boost_same_dialect and filter_dialect and sample_dialect == filter_dialect:
                    similarity = min(100, base_similarity + 20)  # Boost +20%
                elif boost_same_dialect and self.input_dialect and sample_dialect == self.input_dialect:
                    similarity = min(100, base_similarity + 20)  # Boost +20%
                
                # Filter theo dialect nếu được yêu cầu
                if filter_dialect and sample_dialect != filter_dialect:
                    continue
                
                speaker_id = sample['speaker']
                speaker_name = "Unknown"
                if self.speaker_db is not None:
                    speaker_info = self.speaker_db[self.speaker_db['speaker_id'] == speaker_id]
                    if not speaker_info.empty:
                        speaker_name = speaker_info.iloc[0]['vietnamese_name']
                    else:
                        speaker_info = self.speaker_db[self.speaker_db['dialect'] == speaker_id]
                        if not speaker_info.empty:
                            speaker_name = speaker_info.iloc[0]['vietnamese_name']
                
                # Lấy features của sample này
                sample_features = {}
                for col in feature_cols:
                    sample_features[col] = sample.get(col, 0.0)
                
                results.append({
                    'rank': len(results) + 1,
                    'audio_name': sample['audio_name'],
                    'speaker_id': speaker_id,
                    'speaker_name': speaker_name,
                    'similarity': round(similarity, 2),
                    'base_similarity': round(base_similarity, 2),  # Similarity gốc
                    'distance': float(dist),
                    'dialect': sample_dialect,
                    'features': sample_features  # Thêm features để so sánh
                })
                
                # Dừng khi đủ kết quả
                if len(results) >= k:
                    break
            
            # Trả về cả features của input để so sánh
            return results, features
        
        except Exception as e:
            error_msg = f"Lỗi trong search_similar_voices: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise Exception(error_msg)


class VoiceSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Tìm kiếm Giọng nói Tương tự - Enhanced")
        self.root.geometry("1300x950")
        self.root.resizable(True, True)
        
        # Cải thiện giao diện với màu nền nhẹ
        self.root.configure(bg='#f5f5f5')
        
        # Khởi tạo engine
        self.voice_engine = VoiceSearchEngine()
        
        # Biến
        self.audio_file_path = StringVar()
        self.k_value = IntVar(value=10)
        self.model_info = None
        self.current_input_features = None
        self.current_results = None
        self.playing_audio = None
        self.recording = False
        self.recorded_file = None
        self.input_dialect = None  # Dialect của input audio
        self.debug_mode = BooleanVar(value=False)
        
        # Tạo giao diện
        self.create_widgets()
        
        # Load model nếu có
        self.load_model_info()
    
    def create_widgets(self):
        """Tạo các widget cho giao diện với thiết kế đẹp hơn"""
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame chính với padding và màu nền
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(W, E, N, S))
        main_frame.configure(style='Main.TFrame')
        
        # Cấu hình grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title với font đẹp hơn
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=3, pady=(0, 25), sticky=(W, E))
        
        title_label = Label(title_frame, text="🎤 Tìm kiếm Giọng nói Tương tự", 
                            font=("Segoe UI", 20, "bold"), bg='#f5f5f5', fg='#2c3e50')
        title_label.pack()
        
        subtitle_label = Label(title_frame, text="Ứng dụng tìm kiếm giọng nói tương tự bằng AI", 
                              font=("Segoe UI", 10), bg='#f5f5f5', fg='#7f8c8d')
        subtitle_label.pack()
        
        # Frame Model Info với style đẹp hơn
        model_frame = ttk.LabelFrame(main_frame, text="📊 Thông tin Model", padding="12")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(W, E), pady=(0, 15))
        model_frame.columnconfigure(1, weight=1)
        
        self.model_status_label = Label(model_frame, text="Chưa có model", 
                                       foreground="red", font=("Segoe UI", 9, "bold"), bg='#f5f5f5')
        self.model_status_label.grid(row=0, column=0, columnspan=2, sticky=W, pady=(0, 5))
        
        self.model_info_text = ScrolledText(model_frame, height=4, width=70, 
                                            state=DISABLED, font=("Consolas", 9),
                                            bg='#ffffff', fg='#2c3e50', relief=FLAT)
        self.model_info_text.grid(row=1, column=0, columnspan=2, sticky=(W, E), pady=(5, 0))
        
        # Buttons cho model với style đẹp hơn
        model_btn_frame = ttk.Frame(model_frame)
        model_btn_frame.grid(row=2, column=0, columnspan=2, pady=(12, 0))
        
        train_btn = ttk.Button(model_btn_frame, text="🚀 Train Model", 
                               command=self.train_model, width=18)
        train_btn.pack(side=LEFT, padx=5)
        
        load_btn = ttk.Button(model_btn_frame, text="📂 Load Model", 
                             command=self.load_model_info, width=18)
        load_btn.pack(side=LEFT, padx=5)
        
        # Frame File Selection
        file_frame = ttk.LabelFrame(main_frame, text="🎵 Chọn File Audio / Ghi âm", padding="12")
        file_frame.grid(row=2, column=0, columnspan=3, sticky=(W, E), pady=(0, 15))
        file_frame.columnconfigure(1, weight=1)
        
        Label(file_frame, text="File:", font=("Segoe UI", 9), bg='#f5f5f5').grid(row=0, column=0, sticky=W, padx=(0, 8))
        file_entry = Entry(file_frame, textvariable=self.audio_file_path, width=55, 
                          state="readonly", font=("Consolas", 9), bg='#ffffff', relief=FLAT)
        file_entry.grid(row=0, column=1, sticky=(W, E), padx=(0, 8))
        
        select_btn = ttk.Button(file_frame, text="📁 Chọn File", 
                               command=self.select_audio_file, width=15)
        select_btn.grid(row=0, column=2, padx=2)
        
        # Buttons cho audio với icon
        audio_btn_frame = ttk.Frame(file_frame)
        audio_btn_frame.grid(row=1, column=0, columnspan=3, pady=(12, 0))
        
        self.play_input_btn = ttk.Button(audio_btn_frame, text="▶ Phát", 
                                         command=self.play_input_audio, state=DISABLED, width=12)
        self.play_input_btn.pack(side=LEFT, padx=5)
        
        self.stop_audio_btn = ttk.Button(audio_btn_frame, text="⏹ Dừng", 
                                         command=self.stop_audio, state=DISABLED, width=12)
        self.stop_audio_btn.pack(side=LEFT, padx=5)
        
        self.record_btn = ttk.Button(audio_btn_frame, text="● Ghi âm", 
                                    command=self.toggle_record, 
                                    state=NORMAL if SOUNDDEVICE_AVAILABLE else DISABLED, width=12)
        self.record_btn.pack(side=LEFT, padx=5)
        
        self.record_status_label = Label(audio_btn_frame, text="", foreground="red", 
                                        font=("Segoe UI", 9), bg='#f5f5f5')
        self.record_status_label.pack(side=LEFT, padx=15)
        
        # Frame Settings
        settings_frame = ttk.LabelFrame(main_frame, text="⚙️ Cài đặt", padding="12")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(W, E), pady=(0, 15))
        
        Label(settings_frame, text="Số kết quả (K):", font=("Segoe UI", 9), bg='#f5f5f5').grid(row=0, column=0, sticky=W, padx=(0, 10))
        k_spinbox = ttk.Spinbox(settings_frame, from_=1, to=50, textvariable=self.k_value, width=12)
        k_spinbox.grid(row=0, column=1, sticky=W, padx=(0, 25))
        
        # Filter theo dialect
        self.filter_dialect = BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Chỉ tìm cùng vùng miền", 
                       variable=self.filter_dialect).grid(row=0, column=2, sticky=W, padx=(0, 10))
        
        self.dialect_var = StringVar(value="Tất cả")
        Label(settings_frame, text="Vùng miền:", font=("Segoe UI", 9), bg='#f5f5f5').grid(row=1, column=0, sticky=W, padx=(0, 10), pady=(8, 0))
        dialect_combo = ttk.Combobox(settings_frame, textvariable=self.dialect_var, 
                                    values=["Tất cả", "Bắc", "Trung", "Nam"], 
                                    state="readonly", width=14)
        dialect_combo.grid(row=1, column=1, sticky=W, pady=(8, 0))
        dialect_combo.bind('<<ComboboxSelected>>', lambda e: self.update_dialect_filter())
        
        # Boost cùng dialect
        self.boost_same_dialect = BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Ưu tiên cùng vùng miền (+20% similarity)", 
                       variable=self.boost_same_dialect).grid(row=1, column=2, sticky=W, pady=(8, 0))
        
        # Button Search với style đẹp hơn
        search_btn_frame = ttk.Frame(main_frame)
        search_btn_frame.grid(row=4, column=0, columnspan=3, pady=15)
        
        search_btn = ttk.Button(search_btn_frame, text="🔍 Tìm kiếm", 
                               command=self.search_voices, state=DISABLED, width=25)
        search_btn.pack()
        self.search_btn = search_btn
        
        # Progress bar
        self.progress_var = StringVar(value="")
        self.progress_label = Label(main_frame, textvariable=self.progress_var, 
                                   font=("Segoe UI", 9), foreground='#3498db', bg='#f5f5f5')
        self.progress_label.grid(row=5, column=0, columnspan=3, pady=(0, 5))
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate', length=400)
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky=(W, E), pady=(5, 15))
        
        # Frame Results
        results_frame = ttk.LabelFrame(main_frame, text="📋 Kết quả Tìm kiếm", padding="12")
        results_frame.grid(row=7, column=0, columnspan=3, sticky=(W, E, N, S), pady=(0, 15))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        # Treeview cho kết quả
        columns = ('Rank', 'Similarity', 'Speaker', 'Audio Name', 'Dialect', 'Link')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=12)
        
        # Cấu hình cột
        for col in columns:
            self.results_tree.heading(col, text=col)
            if col == 'Rank':
                self.results_tree.column(col, width=60, anchor=CENTER)
            elif col == 'Similarity':
                self.results_tree.column(col, width=110, anchor=CENTER)
            elif col == 'Speaker':
                self.results_tree.column(col, width=140)
            elif col == 'Dialect':
                self.results_tree.column(col, width=90, anchor=CENTER)
            elif col == 'Link':
                self.results_tree.column(col, width=180, anchor=CENTER)
            else:
                self.results_tree.column(col, width=280)
        
        # Bind click vào cột Link để mở folder
        self.results_tree.bind('<Button-1>', self.on_result_click)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(W, E, N, S))
        scrollbar.grid(row=0, column=1, sticky=(N, S))
        
        # Frame Features Comparison
        features_frame = ttk.LabelFrame(main_frame, text="📊 So sánh Thuộc tính", padding="12")
        features_frame.grid(row=8, column=0, columnspan=3, sticky=(W, E, N, S), pady=(0, 10))
        features_frame.columnconfigure(0, weight=1)
        features_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=1)
        
        # Canvas với scrollbar cho features
        features_canvas_frame = ttk.Frame(features_frame)
        features_canvas_frame.grid(row=0, column=0, sticky=(W, E, N, S))
        features_canvas_frame.columnconfigure(0, weight=1)
        features_canvas_frame.rowconfigure(0, weight=1)
        
        self.features_canvas = Canvas(features_canvas_frame, height=220, bg='#ffffff', 
                                      highlightthickness=1, highlightbackground='#bdc3c7')
        features_scrollbar = ttk.Scrollbar(features_canvas_frame, orient=VERTICAL, 
                                          command=self.features_canvas.yview)
        self.features_scrollable_frame = ttk.Frame(self.features_canvas)
        
        self.features_canvas.create_window((0, 0), window=self.features_scrollable_frame, anchor="nw")
        self.features_canvas.configure(yscrollcommand=features_scrollbar.set)
        
        self.features_canvas.grid(row=0, column=0, sticky=(W, E, N, S))
        features_scrollbar.grid(row=0, column=1, sticky=(N, S))
        
        self.features_scrollable_frame.bind("<Configure>", 
            lambda e: self.features_canvas.configure(scrollregion=self.features_canvas.bbox("all")))
        
        self.features_info_label = Label(features_frame, text="Chưa có dữ liệu để so sánh", 
                                         foreground="gray", font=("Segoe UI", 9), bg='#f5f5f5')
        self.features_info_label.grid(row=1, column=0, pady=8)
    
    def load_model_info(self):
        """Load thông tin model"""
        try:
            self.model_info = self.voice_engine.load_trained_model()
            
            if self.model_info:
                # Kiểm tra model đã load đầy đủ chưa
                model_ready = (self.voice_engine.knn_model is not None and 
                              self.voice_engine.scaler is not None and
                              self.voice_engine.df_train is not None and
                              self.voice_engine.feature_columns is not None)
                
                if model_ready:
                    self.model_status_label.config(text="✓ Model đã được load và sẵn sàng", foreground="green")
                    info_text = f"Trained at: {self.model_info['trained_at']}\n"
                    info_text += f"Số samples: {self.model_info['num_samples']}\n"
                    info_text += f"Số features: {self.model_info['num_features']}\n"
                    info_text += f"K neighbors: {self.model_info['k_neighbors']}\n"
                    info_text += f"Model status: ✓ Sẵn sàng"
                    
                    self.model_info_text.config(state=NORMAL)
                    self.model_info_text.delete(1.0, END)
                    self.model_info_text.insert(1.0, info_text)
                    self.model_info_text.config(state=DISABLED)
                    
                    self.search_btn.config(state=NORMAL)
                else:
                    self.model_status_label.config(text="⚠ Model chưa load đầy đủ", foreground="orange")
                    info_text = "Model file tồn tại nhưng chưa load đầy đủ.\nVui lòng train lại model."
                    self.model_info_text.config(state=NORMAL)
                    self.model_info_text.delete(1.0, END)
                    self.model_info_text.insert(1.0, info_text)
                    self.model_info_text.config(state=DISABLED)
                    self.search_btn.config(state=DISABLED)
            else:
                self.model_status_label.config(text="Chưa có model", foreground="red")
                info_text = "Chưa có model được train.\nNhấn 'Train Model' để bắt đầu."
                self.model_info_text.config(state=NORMAL)
                self.model_info_text.delete(1.0, END)
                self.model_info_text.insert(1.0, info_text)
                self.model_info_text.config(state=DISABLED)
                self.search_btn.config(state=DISABLED)
        except Exception as e:
            self.model_status_label.config(text=f"Lỗi khi load model: {str(e)}", foreground="red")
            self.search_btn.config(state=DISABLED)
    
    def train_model(self):
        """Train model trong thread riêng"""
        if messagebox.askyesno("Xác nhận", "Train model có thể mất nhiều thời gian. Bạn có muốn tiếp tục?"):
            self.progress_bar.start()
            self.progress_var.set("Đang train model...")
            self.search_btn.config(state=DISABLED)
            
            def train_thread():
                try:
                    def progress_callback(msg):
                        self.progress_var.set(msg)
                        self.root.update()
                    
                    self.voice_engine.train_model(progress_callback)
                    self.root.after(0, self.on_train_complete, True, "Train model thành công!")
                except Exception as e:
                    self.root.after(0, self.on_train_complete, False, f"Lỗi: {str(e)}")
            
            threading.Thread(target=train_thread, daemon=True).start()
    
    def on_train_complete(self, success, message):
        """Callback khi train xong"""
        self.progress_bar.stop()
        self.progress_var.set("")
        
        if success:
            messagebox.showinfo("Thành công", message)
            self.load_model_info()
        else:
            messagebox.showerror("Lỗi", message)
            self.search_btn.config(state=NORMAL if self.model_info else DISABLED)
    
    def select_audio_file(self):
        """Chọn file audio"""
        file_path = filedialog.askopenfilename(
            title="Chọn file audio",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.webm"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            if allowed_file(file_path):
                self.audio_file_path.set(file_path)
                self.play_input_btn.config(state=NORMAL if PYGAME_AVAILABLE else DISABLED)
            else:
                messagebox.showerror("Lỗi", "Định dạng file không được hỗ trợ!")
    
    def play_input_audio(self):
        """Phát audio input"""
        audio_file = self.audio_file_path.get()
        if audio_file and os.path.exists(audio_file):
            self.play_audio_file(audio_file)
    
    def play_audio_file(self, audio_file):
        """Phát file audio trong thread riêng để không block UI"""
        if not PYGAME_AVAILABLE:
            messagebox.showwarning("Cảnh báo", "Pygame chưa được cài đặt. Không thể phát audio.")
            return
        
        if not os.path.exists(audio_file):
            messagebox.showerror("Lỗi", f"File không tồn tại: {audio_file}")
            return
        
        # Chạy trong thread riêng để không block UI
        def play_thread():
            try:
                # Kiểm tra file tồn tại và có thể đọc
                if not os.path.exists(audio_file):
                    self.root.after(0, lambda: messagebox.showerror("Lỗi", f"File không tồn tại: {audio_file}"))
                    return
                
                # Kiểm tra extension
                if not allowed_file(audio_file):
                    self.root.after(0, lambda: messagebox.showwarning("Cảnh báo", 
                        f"Định dạng file có thể không được hỗ trợ: {audio_file}"))
                
                # Dừng audio đang phát
                if PYGAME_AVAILABLE:
                    try:
                        pygame.mixer.music.stop()
                    except:
                        pass
                
                # Phát audio mới với timeout
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    
                    self.root.after(0, lambda: setattr(self, 'playing_audio', audio_file))
                    self.root.after(0, lambda: self.stop_audio_btn.config(state=NORMAL))
                    
                    print(f"Đang phát: {audio_file}")
                except pygame.error as e:
                    error_msg = f"Lỗi pygame khi phát audio: {str(e)}\nFile: {audio_file}"
                    print(error_msg)
                    self.root.after(0, lambda: messagebox.showerror("Lỗi", error_msg))
                except Exception as e:
                    error_msg = f"Không thể phát audio: {str(e)}\nFile: {audio_file}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    self.root.after(0, lambda: messagebox.showerror("Lỗi", error_msg))
                    
            except Exception as e:
                error_msg = f"Lỗi không mong đợi: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Lỗi", error_msg))
        
        threading.Thread(target=play_thread, daemon=True).start()
    
    def stop_audio(self):
        """Dừng audio đang phát"""
        if PYGAME_AVAILABLE:
            pygame.mixer.music.stop()
        self.playing_audio = None
        self.stop_audio_btn.config(state=DISABLED)
    
    def toggle_record(self):
        """Bật/tắt ghi âm"""
        if not SOUNDDEVICE_AVAILABLE:
            messagebox.showwarning("Cảnh báo", "Sounddevice chưa được cài đặt. Không thể ghi âm.")
            return
        
        if not self.recording:
            # Bắt đầu ghi âm
            self.recording = True
            self.record_btn.config(text="⏸ Dừng ghi", state=NORMAL)
            self.record_status_label.config(text="Đang ghi âm...", foreground="red")
            
            def record_thread():
                try:
                    sample_rate = 44100
                    self.recorded_audio = []
                    
                    # Ghi âm bằng stream để có thể dừng được
                    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
                        while self.recording:
                            chunk, overflowed = stream.read(1024)
                            if overflowed:
                                print("Warning: Audio buffer overflow")
                            self.recorded_audio.append(chunk)
                    
                    # Ghép các chunk lại
                    if self.recorded_audio:
                        self.recorded_audio = np.concatenate(self.recorded_audio, axis=0)
                        
                        # Lưu file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.recorded_file = f"recorded_{timestamp}.wav"
                        sf.write(self.recorded_file, self.recorded_audio, sample_rate)
                        self.root.after(0, self.on_record_complete, True, self.recorded_file)
                    else:
                        self.root.after(0, self.on_record_complete, False, "Không có dữ liệu ghi âm")
                except Exception as e:
                    self.root.after(0, self.on_record_complete, False, str(e))
            
            threading.Thread(target=record_thread, daemon=True).start()
        else:
            # Dừng ghi âm
            self.recording = False
            sd.stop()
            self.record_btn.config(text="● Ghi âm", state=NORMAL)
            self.record_status_label.config(text="Đang lưu...", foreground="orange")
    
    def on_record_complete(self, success, result):
        """Callback khi ghi âm xong"""
        self.recording = False
        self.record_btn.config(text="● Ghi âm", state=NORMAL)
        
        if success:
            self.record_status_label.config(text=f"Đã lưu: {result}", foreground="green")
            self.audio_file_path.set(result)
            self.play_input_btn.config(state=NORMAL if PYGAME_AVAILABLE else DISABLED)
            messagebox.showinfo("Thành công", f"Đã ghi âm và lưu: {result}")
        else:
            self.record_status_label.config(text="Lỗi khi ghi âm", foreground="red")
            messagebox.showerror("Lỗi", f"Lỗi khi ghi âm: {result}")
    
    def open_folder_containing_file(self, file_path):
        """Mở folder chứa file bằng Windows Explorer"""
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File không tồn tại: {file_path}")
            
            folder_path = os.path.dirname(file_path)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder không tồn tại: {folder_path}")
            
            if sys.platform.startswith('win'):
                # Windows: mở folder và highlight file (sử dụng os.path.normpath để xử lý đường dẫn)
                normalized_path = os.path.normpath(file_path)
                # Sử dụng shell=True để xử lý đường dẫn có khoảng trắng tốt hơn
                subprocess.Popen(f'explorer /select,"{normalized_path}"', shell=True)
            elif sys.platform == 'darwin':  # macOS
                subprocess.call(['open', '-R', file_path])
            else:  # Linux
                subprocess.call(['xdg-open', folder_path])
            print(f"Đã mở folder chứa file: {file_path}")
        except Exception as e:
            error_msg = f"Không thể mở folder: {str(e)}\nFile: {file_path}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            messagebox.showerror("Lỗi", error_msg)
    
    def find_audio_file_path(self, audio_name):
        """Tìm đường dẫn file audio từ tên file - ưu tiên trainset"""
        if not audio_name:
            return None
        
        # Thử các cách tìm file - ưu tiên trainset trước
        search_paths = [
            Path("trainset"),
            Path("super_metadata"),
            Path("super_metadata/male_only"),
            Path("new_audio_test"),
            Path("clean_testset"),
            Path("noisy_testset"),
        ]
        
        # Tìm chính xác tên file
        for search_dir in search_paths:
            if not search_dir.exists():
                continue
            
            # Thử tìm với extension trước (tìm chính xác)
            for ext in ['.wav', '.mp3', '.m4a', '.flac', '']:
                test_path = search_dir / f"{audio_name}{ext}"
                if test_path.exists() and test_path.is_file():
                    return str(test_path)
            
            # Tìm đệ quy trong subfolder (chỉ cho trainset)
            if search_dir.name == "trainset":
                for ext in ['.wav', '.mp3', '.m4a', '.flac']:
                    for audio_file in search_dir.rglob(f"{audio_name}{ext}"):
                        if audio_file.is_file():
                            return str(audio_file)
                    # Tìm với pattern (nếu không tìm thấy chính xác)
                    for audio_file in search_dir.rglob("*"):
                        if audio_file.is_file() and audio_file.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']:
                            file_name = audio_file.stem  # Tên không có extension
                            if audio_name in file_name or file_name in audio_name:
                                return str(audio_file)
            else:
                # Tìm với pattern (nếu không tìm thấy chính xác) - chỉ trong folder gốc
                for audio_file in search_dir.glob("*"):
                    if audio_file.is_file():
                        file_name = audio_file.stem  # Tên không có extension
                        if audio_name in file_name or file_name in audio_name:
                            return str(audio_file)
        
        return None
    
    def on_result_click(self, event):
        """Khi click vào kết quả, kiểm tra xem có click vào cột Link không"""
        region = self.results_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        
        column = self.results_tree.identify_column(event.x)
        if not column:
            return
        
        # Lấy index cột (bắt đầu từ #1)
        column_index = int(column.replace('#', '')) - 1
        columns = ('Rank', 'Similarity', 'Speaker', 'Audio Name', 'Dialect', 'Link')
        
        # Chỉ xử lý khi click vào cột Link
        if column_index < len(columns) and columns[column_index] == 'Link':
            item = self.results_tree.identify_row(event.y)
            if not item:
                return
            
            try:
                values = self.results_tree.item(item, 'values')
                if len(values) == 0:
                    return
                
                rank = int(values[0]) - 1
                if not self.current_results or rank >= len(self.current_results):
                    return
                
                result = self.current_results[rank]
                audio_name = result.get('audio_name', '')
                
                if not audio_name:
                    messagebox.showwarning("Cảnh báo", "Không có tên file audio")
                    return
                
                # Tìm và mở folder trong thread riêng
                def find_and_open():
                    audio_path = self.find_audio_file_path(audio_name)
                    
                    if audio_path:
                        print(f"Tìm thấy file: {audio_path}")
                        self.root.after(0, lambda path=audio_path: self.open_folder_containing_file(path))
                    else:
                        error_msg = f"Không tìm thấy file audio: {audio_name}"
                        print(error_msg)
                        self.root.after(0, lambda: messagebox.showwarning("Cảnh báo", error_msg))
                
                threading.Thread(target=find_and_open, daemon=True).start()
                
            except Exception as e:
                error_msg = f"Lỗi khi mở folder: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                messagebox.showerror("Lỗi", error_msg)
    
    def display_features_comparison(self, input_features, selected_result=None):
        """Hiển thị so sánh features với bảng 2 cột và highlight màu đỏ các thuộc tính giống nhau"""
        # Xóa nội dung cũ
        for widget in self.features_scrollable_frame.winfo_children():
            widget.destroy()
        
        if not input_features or not self.current_results:
            self.features_info_label.config(text="Chưa có dữ liệu để so sánh", foreground="gray")
            return
        
        # Nếu chưa chọn result, dùng result đầu tiên
        if selected_result is None and self.current_results:
            selected_result = self.current_results[0]
        
        if not selected_result:
            return
        
        result_features = selected_result.get('features', {})
        similarity = selected_result.get('similarity', 0)
        
        self.features_info_label.config(
            text=f"So sánh với: {selected_result['audio_name']} (Similarity: {similarity:.2f}%)",
            foreground="#3498db"
        )
        
        # Tạo bảng so sánh với 2 cột chính
        row = 0
        
        # Header với style đẹp hơn
        header_bg = '#2c3e50'
        header_fg = '#ffffff'
        
        # Cột 1: Tên thuộc tính
        Label(self.features_scrollable_frame, text="Thuộc tính", 
              font=("Segoe UI", 10, "bold"), bg=header_bg, fg=header_fg, 
              padx=12, pady=6, width=25, anchor=W).grid(row=row, column=0, padx=1, pady=1, sticky=(W, E))
        
        # Cột 2: Input
        Label(self.features_scrollable_frame, text="Input", 
              font=("Segoe UI", 10, "bold"), bg=header_bg, fg=header_fg,
              padx=12, pady=6, width=18, anchor=CENTER).grid(row=row, column=1, padx=1, pady=1, sticky=(W, E))
        
        # Cột 3: Kết quả
        Label(self.features_scrollable_frame, text="Kết quả", 
              font=("Segoe UI", 10, "bold"), bg=header_bg, fg=header_fg,
              padx=12, pady=6, width=18, anchor=CENTER).grid(row=row, column=2, padx=1, pady=1, sticky=(W, E))
        
        # Cột 4: Độ giống
        Label(self.features_scrollable_frame, text="Độ giống", 
              font=("Segoe UI", 10, "bold"), bg=header_bg, fg=header_fg,
              padx=12, pady=6, width=12, anchor=CENTER).grid(row=row, column=3, padx=1, pady=1, sticky=(W, E))
        row += 1
        
        # So sánh từng feature
        feature_cols = self.voice_engine.get_feature_columns() if self.voice_engine.feature_columns else []
        
        for feature_name in sorted(feature_cols):
            input_val = input_features.get(feature_name, 0.0)
            result_val = result_features.get(feature_name, 0.0)
            
            # Tính độ giống - cải thiện logic
            if input_val == 0 and result_val == 0:
                similarity_pct = 100.0
            elif abs(input_val) < 1e-10 and abs(result_val) < 1e-10:
                similarity_pct = 100.0
            else:
                # Sử dụng relative error
                avg_val = (abs(input_val) + abs(result_val)) / 2
                if avg_val > 1e-10:
                    diff = abs(input_val - result_val)
                    similarity_pct = max(0, (1 - diff / avg_val) * 100)
                else:
                    similarity_pct = 0.0
            
            # Xác định màu sắc: đỏ đậm nếu giống > 85%, đỏ nhạt nếu > 70%, vàng nếu > 50%
            is_similar = similarity_pct > 70  # Ngưỡng để coi là giống
            
            if similarity_pct > 85:
                bg_color = "#ff9999"  # Đỏ đậm
                text_color = "#cc0000"  # Đỏ đậm cho text
                is_highlight = True
            elif similarity_pct > 70:
                bg_color = "#ffcccc"  # Đỏ nhạt
                text_color = "#cc0000"  # Đỏ cho text
                is_highlight = True
            elif similarity_pct > 50:
                bg_color = "#ffffcc"  # Vàng nhạt
                text_color = "#000000"  # Đen
                is_highlight = False
            else:
                bg_color = "#ffffff"  # Trắng
                text_color = "#000000"  # Đen
                is_highlight = False
            
            # Tên feature
            font_weight = "bold" if is_highlight else "normal"
            Label(self.features_scrollable_frame, text=feature_name, 
                 bg=bg_color, fg=text_color, font=("Segoe UI", 9, font_weight), 
                 padx=12, pady=4, anchor=W, width=25).grid(
                 row=row, column=0, padx=1, pady=1, sticky=W)
            
            # Giá trị input - highlight nếu giống
            Label(self.features_scrollable_frame, text=f"{input_val:.6f}", 
                 bg=bg_color, fg=text_color, font=("Consolas", 9, font_weight), 
                 padx=12, pady=4, anchor=E, width=18).grid(
                 row=row, column=1, padx=1, pady=1, sticky=E)
            
            # Giá trị result - highlight nếu giống
            Label(self.features_scrollable_frame, text=f"{result_val:.6f}", 
                 bg=bg_color, fg=text_color, font=("Consolas", 9, font_weight), 
                 padx=12, pady=4, anchor=E, width=18).grid(
                 row=row, column=2, padx=1, pady=1, sticky=E)
            
            # Độ giống % - màu đỏ nếu giống
            similarity_color = "#cc0000" if is_highlight else "#000000"
            Label(self.features_scrollable_frame, text=f"{similarity_pct:.1f}%", 
                 bg=bg_color, fg=similarity_color, 
                 font=("Segoe UI", 9, "bold"), padx=12, pady=4, anchor=CENTER, width=12).grid(
                 row=row, column=3, padx=1, pady=1, sticky=CENTER)
            
            row += 1
        
        # Cập nhật scroll region
        self.features_scrollable_frame.update_idletasks()
        self.features_canvas.configure(scrollregion=self.features_canvas.bbox("all"))
    
    def search_voices(self):
        """Tìm kiếm giọng nói tương tự"""
        audio_file = self.audio_file_path.get()
        
        if not audio_file or not os.path.exists(audio_file):
            messagebox.showerror("Lỗi", "Vui lòng chọn file audio hợp lệ!")
            return
        
        if not self.model_info:
            messagebox.showerror("Lỗi", "Chưa có model. Vui lòng train model trước!")
            return
        
        # Clear kết quả cũ
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Disable button và hiển thị progress
        self.search_btn.config(state=DISABLED)
        self.progress_bar.start()
        self.progress_var.set("Đang xử lý...")
        
        def search_thread():
            try:
                def progress_callback(msg):
                    self.progress_var.set(msg)
                    self.root.update()
                
                # Kiểm tra lại model trước khi search
                if not self.model_info:
                    raise Exception("Model chưa được load. Vui lòng train hoặc load model trước!")
                
                if self.voice_engine.knn_model is None:
                    raise Exception("KNN model chưa được load. Vui lòng train hoặc load model trước!")
                
                k = self.k_value.get()
                
                # Lấy filter dialect
                filter_dialect = None
                selected_dialect = self.dialect_var.get()
                if selected_dialect != "Tất cả":
                    filter_dialect = selected_dialect
                
                # Boost cùng dialect
                boost_same = self.boost_same_dialect.get()
                
                results, input_features = self.voice_engine.search_similar_voices(
                    audio_file, k, progress_callback, 
                    filter_dialect=filter_dialect,
                    boost_same_dialect=boost_same
                )
                self.root.after(0, self.on_search_complete, True, results, input_features)
            except Exception as e:
                import traceback
                error_detail = f"{str(e)}\n\nChi tiết:\n{traceback.format_exc()}"
                self.root.after(0, self.on_search_complete, False, None, None, error_detail)
        
        threading.Thread(target=search_thread, daemon=True).start()
    
    def on_search_complete(self, success, results=None, input_features=None, error=None):
        """Callback khi search xong"""
        self.progress_bar.stop()
        self.progress_var.set("")
        self.search_btn.config(state=NORMAL)
        
        if success and results:
            # Lưu kết quả và features
            self.current_results = results
            self.current_input_features = input_features
            
            # Hiển thị kết quả
            for result in results:
                similarity = result['similarity']
                color_tag = 'high' if similarity > 80 else 'medium' if similarity > 60 else 'low'
                
                # Tạo link text với icon folder
                audio_name = result['audio_name']
                link_text = f"📁 Mở folder"
                
                self.results_tree.insert('', END, values=(
                    result['rank'],
                    f"{similarity:.2f}%",
                    result['speaker_name'],
                    result['audio_name'],
                    result['dialect'],
                    link_text
                ), tags=(color_tag, 'link'))
            
            # Tag colors
            self.results_tree.tag_configure('high', foreground='#27ae60')
            self.results_tree.tag_configure('medium', foreground='#f39c12')
            self.results_tree.tag_configure('low', foreground='#e74c3c')
            
            # Tag cho link - màu xanh
            self.results_tree.tag_configure('link', foreground='#3498db')
            
            # Hiển thị so sánh features với kết quả đầu tiên
            if input_features:
                self.display_features_comparison(input_features, results[0] if results else None)
            
            # Bind selection change để cập nhật features comparison
            self.results_tree.bind('<<TreeviewSelect>>', self.on_result_select)
            
            messagebox.showinfo("Thành công", f"Tìm thấy {len(results)} kết quả!")
        else:
            # Hiển thị lỗi chi tiết hơn
            error_msg = error or "Có lỗi xảy ra khi tìm kiếm!"
            # Tạo window hiển thị lỗi chi tiết
            error_window = Toplevel(self.root)
            error_window.title("Lỗi")
            error_window.geometry("600x400")
            error_window.configure(bg='#f5f5f5')
            
            error_text = ScrolledText(error_window, wrap=WORD, width=70, height=20,
                                     font=("Consolas", 9), bg='#ffffff', fg='#2c3e50')
            error_text.pack(fill=BOTH, expand=True, padx=10, pady=10)
            error_text.insert(1.0, error_msg)
            error_text.config(state=DISABLED)
            
            ttk.Button(error_window, text="Đóng", command=error_window.destroy).pack(pady=10)
    
    def update_dialect_filter(self):
        """Cập nhật filter dialect"""
        # Có thể thêm logic tự động detect dialect từ input audio ở đây
        pass
    
    def on_result_select(self, event):
        """Khi chọn một kết quả, cập nhật features comparison"""
        selection = self.results_tree.selection()
        if selection and self.current_input_features and self.current_results:
            item = self.results_tree.item(selection[0])
            values = item['values']
            if len(values) > 0:
                rank = int(values[0]) - 1
                if rank < len(self.current_results):
                    selected_result = self.current_results[rank]
                    self.display_features_comparison(self.current_input_features, selected_result)


def main():
    """Hàm main"""
    root = Tk()
    app = VoiceSearchApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()

