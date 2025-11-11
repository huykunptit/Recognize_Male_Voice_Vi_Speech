#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ứng dụng tìm kiếm giọng nói tương tự sử dụng K-Nearest Neighbors
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
    print("   3. Xem file FIX_NUMBA_ERROR.md để biết thêm chi tiết")
    print("")
    # Sẽ thử import lại khi cần thiết

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

# Set encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['TRAINED_MODEL_FILE'] = 'trained_model.json'
app.config['SCALER_FILE'] = 'scaler.joblib'
app.config['KNN_MODEL_FILE'] = 'knn_model.joblib'

# Tạo folder uploads nếu chưa có
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class VoiceSearchEngine:
    def __init__(self):
        self.super_metadata_folder = "super_metadata/male_only"
        self.speaker_db_file = "speaker_database.csv"
        self.scaler = None
        self.knn_model = None
        self.feature_columns = None
        self.df_train = None
        self.speaker_db = None
        
    def load_speaker_database(self):
        """Load speaker database với tên tiếng Việt"""
        try:
            self.speaker_db = pd.read_csv(self.speaker_db_file, encoding='utf-8')
            print(f"Đã load {len(self.speaker_db)} speakers từ database")
            return True
        except Exception as e:
            print(f"Lỗi khi load speaker database: {e}")
            return False
    
    def load_training_data(self):
        """Load và merge tất cả CSV files từ super_metadata/male_only"""
        print("Đang load dữ liệu training từ super_metadata/male_only...")
        all_data = []
        
        csv_files = sorted(Path(self.super_metadata_folder).glob("*.csv"))
        print(f"Tìm thấy {len(csv_files)} file CSV")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                all_data.append(df)
                print(f"  - Đã load {len(df)} records từ {csv_file.name}")
            except Exception as e:
                print(f"  - Lỗi khi load {csv_file.name}: {e}")
        
        if not all_data:
            raise ValueError("Không tìm thấy dữ liệu training!")
        
        # Merge tất cả dataframes
        self.df_train = pd.concat(all_data, ignore_index=True)
        print(f"\nTổng cộng: {len(self.df_train)} records")
        print(f"Số cột: {len(self.df_train.columns)}")
        
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
        print("\n=== Bắt đầu train model ===")
        
        # Load dữ liệu
        self.load_training_data()
        self.load_speaker_database()
        
        # Lấy feature columns
        feature_cols = self.get_feature_columns()
        print(f"Số features: {len(feature_cols)}")
        
        # Chuẩn bị dữ liệu training
        X_train = self.df_train[feature_cols].fillna(0).values
        
        # Chuẩn hóa dữ liệu
        print("Đang chuẩn hóa dữ liệu...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train KNN model
        print("Đang train KNN model (K=10)...")
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
        self.knn_model.fit(X_train_scaled)
        
        # Lưu model và scaler
        joblib.dump(self.scaler, app.config['SCALER_FILE'])
        joblib.dump(self.knn_model, app.config['KNN_MODEL_FILE'])
        
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
        
        with open(app.config['TRAINED_MODEL_FILE'], 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Đã train xong model!")
        print(f"  - Số samples: {len(self.df_train)}")
        print(f"  - Số features: {len(feature_cols)}")
        print(f"  - Đã lưu model vào: {app.config['TRAINED_MODEL_FILE']}")
        
        return training_info
    
    def load_trained_model(self):
        """Load model đã train từ file"""
        try:
            # Load training info
            with open(app.config['TRAINED_MODEL_FILE'], 'r', encoding='utf-8') as f:
                training_info = json.load(f)
            
            # Load scaler và model
            self.scaler = joblib.load(app.config['SCALER_FILE'])
            self.knn_model = joblib.load(app.config['KNN_MODEL_FILE'])
            
            # Load dữ liệu training
            self.load_training_data()
            self.load_speaker_database()
            self.feature_columns = training_info['feature_columns']
            
            print(f"✓ Đã load model đã train (trained at: {training_info['trained_at']})")
            return training_info
        except Exception as e:
            print(f"Không tìm thấy model đã train: {e}")
            return None
    
    def extract_audio_features(self, audio_path):
        """Trích xuất features từ file audio (giống như trong create_super_metadata.py)"""
        global LIBROSA_AVAILABLE
        try:
            # Kiểm tra librosa có sẵn không
            if not LIBROSA_AVAILABLE:
                try:
                    import librosa
                    LIBROSA_AVAILABLE = True
                except ImportError:
                    raise ImportError("librosa không thể sử dụng. Vui lòng cài đặt lại: pip install --upgrade numba llvmlite librosa")
            
            # Load audio file với error handling tốt hơn
            try:
                y, sr = librosa.load(audio_path, sr=None)
            except Exception as e:
                # Thử lại với duration limit nếu file quá dài
                print(f"Lỗi khi load audio lần đầu: {e}")
                try:
                    y, sr = librosa.load(audio_path, sr=None, duration=60)  # Giới hạn 60s
                except Exception as e2:
                    raise Exception(f"Không thể load file audio: {str(e2)}. Có thể file không đúng định dạng hoặc bị hỏng.")
            
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
            import traceback
            error_msg = f"Lỗi khi trích xuất features từ {audio_path}: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return None
    
    def search_similar_voices(self, audio_path, k=10):
        """Tìm K giọng nói tương tự nhất"""
        try:
            # Kiểm tra model đã được load chưa
            if self.knn_model is None or self.scaler is None:
                raise ValueError("Model chưa được train hoặc load. Vui lòng train model trước!")
            
            # Trích xuất features từ audio
            features = self.extract_audio_features(audio_path)
            if features is None:
                raise ValueError("Không thể trích xuất features từ file audio")
            
            # Chuẩn bị feature vector
            feature_cols = self.get_feature_columns()
            if not feature_cols:
                raise ValueError("Không tìm thấy feature columns. Vui lòng train model lại!")
            
            feature_vector = np.array([features.get(col, 0.0) for col in feature_cols]).reshape(1, -1)
            
            # Chuẩn hóa
            try:
                feature_vector_scaled = self.scaler.transform(feature_vector)
            except Exception as e:
                raise ValueError(f"Lỗi khi chuẩn hóa features: {str(e)}")
            
            # Tìm K nearest neighbors
            try:
                distances, indices = self.knn_model.kneighbors(feature_vector_scaled, n_neighbors=k)
            except Exception as e:
                raise ValueError(f"Lỗi khi tìm kiếm neighbors: {str(e)}")
            
            # Lấy thông tin các samples tương tự
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                sample = self.df_train.iloc[idx]
                
                # Tính similarity percentage (cosine distance -> similarity)
                similarity = (1 - dist) * 100
                
                # Lấy thông tin speaker
                speaker_id = sample['speaker']
                speaker_name = "Unknown"
                if self.speaker_db is not None:
                    # Map speaker từ training data với speaker_id trong database
                    speaker_info = self.speaker_db[self.speaker_db['speaker_id'] == speaker_id]
                    if not speaker_info.empty:
                        speaker_name = speaker_info.iloc[0]['vietnamese_name']
                    else:
                        # Thử tìm với dialect nếu không tìm thấy
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
            import traceback
            error_msg = f"Lỗi trong search_similar_voices: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            raise

# Khởi tạo Voice Search Engine
voice_engine = VoiceSearchEngine()

@app.route('/')
def index():
    """Trang chủ"""
    # Kiểm tra xem đã có model chưa
    model_info = voice_engine.load_trained_model()
    return render_template('index.html', model_info=model_info)

@app.route('/train', methods=['POST'])
def train():
    """Train model từ dữ liệu training"""
    try:
        training_info = voice_engine.train_model()
        return jsonify({
            'success': True,
            'message': 'Đã train model thành công!',
            'training_info': training_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi khi train: {str(e)}'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload và xử lý file audio"""
    print(f"\n=== Upload Request ===")
    print(f"Method: {request.method}")
    print(f"Content-Type: {request.content_type}")
    print(f"Files: {list(request.files.keys())}")
    print(f"Form: {dict(request.form)}")
    
    if 'file' not in request.files:
        print("ERROR: Không có 'file' trong request.files")
        return jsonify({'success': False, 'message': 'Không có file được upload'}), 400
    
    file = request.files['file']
    print(f"File object: {file}")
    print(f"Filename: {file.filename}")
    print(f"Content-Type: {file.content_type}")
    
    if file.filename == '':
        print("ERROR: Filename rỗng")
        return jsonify({'success': False, 'message': 'Chưa chọn file'}), 400
    
    if not allowed_file(file.filename):
        print(f"ERROR: Định dạng không được hỗ trợ: {file.filename}")
        return jsonify({'success': False, 'message': f'Định dạng file không được hỗ trợ: {file.filename.rsplit(".", 1)[-1] if "." in file.filename else "unknown"}'}), 400
    
    # Kiểm tra model đã được train chưa
    if voice_engine.knn_model is None:
        model_info = voice_engine.load_trained_model()
        if model_info is None:
            return jsonify({
                'success': False,
                'message': 'Chưa có model. Vui lòng train model trước!'
            }), 400
    
    try:
        # Lưu file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Đã lưu file: {filepath}")
        print(f"Kích thước file: {os.path.getsize(filepath)} bytes")
        
        # Kiểm tra file tồn tại
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File không tồn tại sau khi lưu'}), 500
        
        # Tìm kiếm giọng nói tương tự
        k = int(request.form.get('k', 10))
        print(f"Bắt đầu tìm kiếm với K={k}")
        
        results = voice_engine.search_similar_voices(filepath, k=k)
        
        if results is None:
            return jsonify({'success': False, 'message': 'Lỗi khi xử lý file audio. Vui lòng kiểm tra định dạng file.'}), 500
        
        print(f"Tìm thấy {len(results)} kết quả")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'results': results
        })
        
    except ValueError as e:
        # Lỗi validation
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        import traceback
        error_msg = f'Lỗi khi xử lý: {str(e)}'
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': error_msg}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Lấy thông tin model"""
    model_info = voice_engine.load_trained_model()
    if model_info:
        return jsonify({'success': True, 'model_info': model_info})
    else:
        return jsonify({'success': False, 'message': 'Chưa có model được train'})

if __name__ == '__main__':
    print("=" * 60)
    print("Ứng dụng Tìm kiếm Giọng nói Tương tự")
    print("=" * 60)
    print("\nĐang khởi động server...")
    print("Truy cập: http://localhost:5000")
    print("\nNhấn Ctrl+C để dừng server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

