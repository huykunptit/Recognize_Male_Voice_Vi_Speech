#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ứng dụng tìm kiếm giọng nói tương tự - Phiên bản CLI
Chạy trực tiếp từ command line, không cần web server
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Xử lý lỗi numba/librosa trên Windows
LIBROSA_AVAILABLE = False
try:
    import librosa
    LIBROSA_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"⚠️  Warning: librosa không thể import: {e}")
    print("💡 Giải pháp:")
    print("   1. Cài đặt Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("   2. Hoặc chạy: pip install --upgrade numba llvmlite librosa")
    print("")
    sys.exit(1)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

warnings.filterwarnings('ignore')

# Set encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass

console = Console()

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
        
    def load_speaker_database(self):
        """Load speaker database với tên tiếng Việt"""
        try:
            self.speaker_db = pd.read_csv(self.speaker_db_file, encoding='utf-8')
            console.print(f"[green]✓[/] Đã load {len(self.speaker_db)} speakers từ database")
            return True
        except Exception as e:
            console.print(f"[red]✗[/] Lỗi khi load speaker database: {e}")
            return False
    
    def load_training_data(self):
        """Load và merge tất cả CSV files từ super_metadata/male_only"""
        console.print("[cyan]Đang load dữ liệu training từ super_metadata/male_only...[/]")
        all_data = []
        
        csv_files = sorted(Path(self.super_metadata_folder).glob("*.csv"))
        console.print(f"Tìm thấy {len(csv_files)} file CSV")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                all_data.append(df)
                console.print(f"  - Đã load {len(df)} records từ {csv_file.name}")
            except Exception as e:
                console.print(f"  - [red]Lỗi[/] khi load {csv_file.name}: {e}")
        
        if not all_data:
            raise ValueError("Không tìm thấy dữ liệu training!")
        
        # Merge tất cả dataframes
        self.df_train = pd.concat(all_data, ignore_index=True)
        console.print(f"\n[green]Tổng cộng:[/] {len(self.df_train)} records")
        console.print(f"[green]Số cột:[/] {len(self.df_train.columns)}")
        
        return True
    
    def get_feature_columns(self):
        """Lấy danh sách các cột features (bỏ qua metadata columns)"""
        if self.feature_columns is None:
            exclude_cols = ['audio_name', 'dialect', 'gender', 'speaker']
            self.feature_columns = [col for col in self.df_train.columns 
                                   if col not in exclude_cols]
        return self.feature_columns
    
    def train_model(self):
        """Train KNN model từ dữ liệu training"""
        console.print("\n[bold cyan]=== Bắt đầu train model ===[/]\n")
        
        # Load dữ liệu
        self.load_training_data()
        self.load_speaker_database()
        
        # Lấy feature columns
        feature_cols = self.get_feature_columns()
        console.print(f"[green]Số features:[/] {len(feature_cols)}")
        
        # Chuẩn bị dữ liệu training
        X_train = self.df_train[feature_cols].fillna(0).values
        
        # Chuẩn hóa dữ liệu
        console.print("[cyan]Đang chuẩn hóa dữ liệu...[/]")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train KNN model
        console.print("[cyan]Đang train KNN model (K=10)...[/]")
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
        self.knn_model.fit(X_train_scaled)
        
        # Lưu model và scaler
        joblib.dump(self.scaler, SCALER_FILE)
        joblib.dump(self.knn_model, KNN_MODEL_FILE)
        
        # Lưu thông tin training vào JSON
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
        
        console.print(f"\n[bold green]✓ Đã train xong model![/]")
        console.print(f"  - Số samples: {len(self.df_train)}")
        console.print(f"  - Số features: {len(feature_cols)}")
        console.print(f"  - Đã lưu model vào: {TRAINED_MODEL_FILE}")
        
        return training_info
    
    def load_trained_model(self):
        """Load model đã train từ file"""
        try:
            # Load training info
            with open(TRAINED_MODEL_FILE, 'r', encoding='utf-8') as f:
                training_info = json.load(f)
            
            # Load scaler và model
            self.scaler = joblib.load(SCALER_FILE)
            self.knn_model = joblib.load(KNN_MODEL_FILE)
            
            # Load dữ liệu training
            self.load_training_data()
            self.load_speaker_database()
            self.feature_columns = training_info['feature_columns']
            
            console.print(f"[green]✓[/] Đã load model đã train (trained at: {training_info['trained_at']})")
            return training_info
        except Exception as e:
            console.print(f"[red]✗[/] Không tìm thấy model đã train: {e}")
            return None
    
    def extract_audio_features(self, audio_path):
        """Trích xuất features từ file audio"""
        global LIBROSA_AVAILABLE
        try:
            if not LIBROSA_AVAILABLE:
                import librosa
                LIBROSA_AVAILABLE = True
            
            # Load audio file
            try:
                y, sr = librosa.load(audio_path, sr=None)
            except Exception as e:
                console.print(f"[yellow]Thử lại với duration limit...[/]")
                y, sr = librosa.load(audio_path, sr=None, duration=60)
            
            features = {}
            
            # 1. Pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            features['pitch_mean'] = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0
            features['pitch_std'] = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0.0
            features['pitch_range'] = float(np.nanmax(f0) - np.nanmin(f0)) if not np.all(np.isnan(f0)) else 0.0
            
            # 2. Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # 3. Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 5. MFCC (13 hệ số)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            # 6. Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            # 7. Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            features['spectral_contrast_std'] = float(np.std(contrast))
            
            # 8. Tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features['tonnetz_mean'] = float(np.mean(tonnetz))
            features['tonnetz_std'] = float(np.std(tonnetz))
            
            # 9. RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            features['rms_max'] = float(np.max(rms))
            features['rms_min'] = float(np.min(rms))
            
            # 10. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # 11. Duration
            features['duration'] = float(len(y) / sr)
            
            # 12. Loudness
            features['loudness'] = float(20 * np.log10(np.mean(np.abs(y)) + 1e-10))
            features['loudness_peak'] = float(20 * np.log10(np.max(np.abs(y)) + 1e-10))
            
            # 13. Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # 14. Spectral Flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            features['spectral_flatness_std'] = float(np.std(spectral_flatness))
            
            # 15. Harmonic-to-noise ratio
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            hnr = np.mean(20 * np.log10(np.abs(y_harmonic) / (np.abs(y_percussive) + 1e-10)))
            features['hnr'] = float(hnr) if not np.isnan(hnr) and not np.isinf(hnr) else 0.0
            
            # 16. Spectral Slope
            spectral_slope = librosa.feature.spectral_slope(y=y)[0]
            features['spectral_slope_mean'] = float(np.mean(spectral_slope))
            features['spectral_slope_std'] = float(np.std(spectral_slope))
            
            # 17. Spectral Kurtosis
            spectral_kurtosis = librosa.feature.spectral_kurtosis(y=y)[0]
            features['spectral_kurtosis_mean'] = float(np.mean(spectral_kurtosis))
            features['spectral_kurtosis_std'] = float(np.std(spectral_kurtosis))
            
            # 18. Spectral Skewness
            spectral_skewness = librosa.feature.spectral_skewness(y=y)[0]
            features['spectral_skewness_mean'] = float(np.mean(spectral_skewness))
            features['spectral_skewness_std'] = float(np.std(spectral_skewness))
            
            # 19. Onset Strength
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength_mean'] = float(np.mean(onset_strength))
            features['onset_strength_std'] = float(np.std(onset_strength))
            
            # 20. Spectral Flux
            spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
            features['spectral_flux'] = float(spectral_flux)
            
            return features
            
        except Exception as e:
            console.print(f"[red]✗[/] Lỗi khi trích xuất features: {e}")
            return None
    
    def search_similar_voices(self, audio_path, k=10):
        """Tìm K giọng nói tương tự nhất"""
        try:
            if self.knn_model is None or self.scaler is None:
                raise ValueError("Model chưa được train hoặc load. Vui lòng train model trước!")
            
            # Trích xuất features
            console.print("[cyan]Đang trích xuất features từ file audio...[/]")
            features = self.extract_audio_features(audio_path)
            if features is None:
                raise ValueError("Không thể trích xuất features từ file audio")
            
            # Chuẩn bị feature vector
            feature_cols = self.get_feature_columns()
            if not feature_cols:
                raise ValueError("Không tìm thấy feature columns. Vui lòng train model lại!")
            
            feature_vector = np.array([features.get(col, 0.0) for col in feature_cols]).reshape(1, -1)
            
            # Chuẩn hóa
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Tìm K nearest neighbors
            console.print(f"[cyan]Đang tìm {k} giọng nói tương tự nhất...[/]")
            distances, indices = self.knn_model.kneighbors(feature_vector_scaled, n_neighbors=k)
            
            # Lấy thông tin các samples tương tự
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                sample = self.df_train.iloc[idx]
                
                # Tính similarity percentage
                similarity = (1 - dist) * 100
                
                # Lấy thông tin speaker
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
                
                results.append({
                    'rank': i + 1,
                    'audio_name': sample['audio_name'],
                    'speaker_id': speaker_id,
                    'speaker_name': speaker_name,
                    'similarity': round(similarity, 2),
                    'distance': float(dist),
                    'dialect': sample.get('dialect', 'N/A')
                })
            
            return results
        
        except Exception as e:
            console.print(f"[red]✗[/] Lỗi trong search_similar_voices: {e}")
            raise


def display_results(results, audio_file):
    """Hiển thị kết quả tìm kiếm"""
    console.print(f"\n[bold cyan]Kết quả tìm kiếm cho:[/] [yellow]{audio_file}[/]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Similarity", style="green", width=12)
    table.add_column("Speaker", style="yellow", width=20)
    table.add_column("Audio Name", style="white", width=40)
    table.add_column("Dialect", style="blue", width=10)
    
    for result in results:
        similarity_color = "green" if result['similarity'] > 80 else "yellow" if result['similarity'] > 60 else "red"
        table.add_row(
            str(result['rank']),
            f"[{similarity_color}]{result['similarity']:.2f}%[/]",
            result['speaker_name'],
            result['audio_name'],
            result['dialect']
        )
    
    console.print(table)


def main():
    """Hàm main"""
    console.print("[bold cyan]Ứng dụng Tìm kiếm Giọng nói Tương tự - CLI[/]\n")
    
    # Khởi tạo engine
    voice_engine = VoiceSearchEngine()
    
    # Parse arguments
    if len(sys.argv) < 2:
        console.print("[yellow]Cách sử dụng:[/]")
        console.print("  python voice_search_cli.py train                    # Train model")
        console.print("  python voice_search_cli.py search <audio_file> [k]  # Tìm kiếm giọng nói tương tự")
        console.print("  python voice_search_cli.py info                      # Xem thông tin model")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'train':
        # Train model
        try:
            voice_engine.train_model()
        except Exception as e:
            console.print(f"[red]✗[/] Lỗi khi train: {e}")
            sys.exit(1)
    
    elif command == 'search':
        # Tìm kiếm
        if len(sys.argv) < 3:
            console.print("[red]✗[/] Thiếu tên file audio!")
            console.print("  python voice_search_cli.py search <audio_file> [k]")
            sys.exit(1)
        
        audio_file = sys.argv[2]
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        
        # Kiểm tra file tồn tại
        if not os.path.exists(audio_file):
            console.print(f"[red]✗[/] File không tồn tại: {audio_file}")
            sys.exit(1)
        
        if not allowed_file(audio_file):
            console.print(f"[red]✗[/] Định dạng file không được hỗ trợ!")
            sys.exit(1)
        
        # Load model
        model_info = voice_engine.load_trained_model()
        if model_info is None:
            console.print("[red]✗[/] Chưa có model. Vui lòng train model trước!")
            console.print("  python voice_search_cli.py train")
            sys.exit(1)
        
        # Tìm kiếm
        try:
            results = voice_engine.search_similar_voices(audio_file, k=k)
            display_results(results, audio_file)
        except Exception as e:
            console.print(f"[red]✗[/] Lỗi: {e}")
            sys.exit(1)
    
    elif command == 'info':
        # Xem thông tin model
        model_info = voice_engine.load_trained_model()
        if model_info:
            console.print(Panel(
                f"[cyan]Trained at:[/] {model_info['trained_at']}\n"
                f"[cyan]Số samples:[/] {model_info['num_samples']}\n"
                f"[cyan]Số features:[/] {model_info['num_features']}\n"
                f"[cyan]Model type:[/] {model_info['model_type']}\n"
                f"[cyan]K neighbors:[/] {model_info['k_neighbors']}\n"
                f"[cyan]Metric:[/] {model_info['metric']}",
                title="[bold green]Model Information[/]",
                border_style="green"
            ))
        else:
            console.print("[red]✗[/] Chưa có model được train")
            sys.exit(1)
    
    else:
        console.print(f"[red]✗[/] Lệnh không hợp lệ: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()

