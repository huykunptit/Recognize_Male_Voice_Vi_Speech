
import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import shutil
from datetime import datetime
import json

class VoiceComparisonApp:
    def __init__(self):
        self.data_check_folder = "data_check"
        self.metadata_folder = "metadata"
        self.super_metadata_folder = "super_metadata"
        self.speaker_db_file = "speaker_database.csv"
        
        # Tạo các folder cần thiết
        self.setup_folders()
        
    def setup_folders(self):
        """Tạo các folder cần thiết"""
        folders = [self.data_check_folder, self.super_metadata_folder]
        for folder in folders:
            Path(folder).mkdir(exist_ok=True)
            print(f"Đã tạo folder: {folder}")
    
    def extract_audio_features(self, audio_path):
        """Trích xuất đặc trưng âm thanh từ file audio"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path)
            
            # Tính toán các đặc trưng
            features = {}
            
            # 1. Độ cao (Pitch) - F0
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            features['pitch_mean'] = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
            features['pitch_std'] = np.nanstd(f0) if not np.all(np.isnan(f0)) else 0
            
            # 2. Độ trầm bổng (Spectral Centroid)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # 3. Độ rõ ràng (Spectral Rolloff)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 5. MFCC (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
            # 6. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # 7. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = np.mean(contrast)
            features['spectral_contrast_std'] = np.std(contrast)
            
            # 8. Tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz)
            features['tonnetz_std'] = np.std(tonnetz)
            
            # 9. RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # 10. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # 11. Duration
            features['duration'] = len(y) / sr
            
            # 12. Loudness (dB)
            features['loudness'] = 20 * np.log10(np.mean(np.abs(y)) + 1e-10)
            
            # 13. Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # 14. Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            features['spectral_flatness_std'] = np.std(spectral_flatness)
            
            # 15. Harmonic-to-noise ratio
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            hnr = np.mean(20 * np.log10(np.abs(y_harmonic) / (np.abs(y_percussive) + 1e-10)))
            features['hnr'] = hnr if not np.isnan(hnr) and not np.isinf(hnr) else 0
            
            return features
            
        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng từ {audio_path}: {e}")
            return None
    
    def upload_audio_file(self, audio_path):
        """Upload file audio vào data_check folder"""
        try:
            # Kiểm tra file có tồn tại không
            if not os.path.exists(audio_path):
                print(f"File {audio_path} không tồn tại!")
                return None
            
            # Tạo tên file mới với timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"upload_{timestamp}_{os.path.basename(audio_path)}"
            dest_path = os.path.join(self.data_check_folder, filename)
            
            # Copy file vào data_check folder
            shutil.copy2(audio_path, dest_path)
            print(f"Đã upload file: {filename}")
            
            # Trích xuất đặc trưng
            features = self.extract_audio_features(dest_path)
            if features:
                # Lưu đặc trưng vào file JSON
                features_file = dest_path.replace('.mp3', '_features.json')
                with open(features_file, 'w', encoding='utf-8') as f:
                    json.dump(features, f, ensure_ascii=False, indent=2)
                print(f"Đã lưu đặc trưng: {features_file}")
            
            return dest_path, features
            
        except Exception as e:
            print(f"Lỗi khi upload file: {e}")
            return None
    
    def compare_with_trainset(self, uploaded_features):
        """So sánh đặc trưng với trainset để tìm giọng tương tự"""
        try:
            # Load trainset metadata
            trainset_path = os.path.join(self.metadata_folder, "trainset.csv")
            if not os.path.exists(trainset_path):
                print("Không tìm thấy file trainset.csv!")
                return None
            
            df = pd.read_csv(trainset_path)
            print(f"Đã load {len(df)} file từ trainset")
            
            # Tính toán similarity (đơn giản bằng Euclidean distance)
            similarities = []
            
            for idx, row in df.iterrows():
                # Tạo path đến file audio
                audio_path = os.path.join("trainset", row['audio_name'])
                
                if os.path.exists(audio_path):
                    # Trích xuất đặc trưng từ file trainset
                    train_features = self.extract_audio_features(audio_path)
                    
                    if train_features:
                        # Tính similarity
                        similarity = self.calculate_similarity(uploaded_features, train_features)
                        similarities.append({
                            'audio_name': row['audio_name'],
                            'speaker': row['speaker'],
                            'dialect': row['dialect'],
                            'similarity': similarity
                        })
            
            # Sắp xếp theo độ tương tự (cao nhất trước)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similarities[:10]  # Trả về top 10 giọng tương tự nhất
            
        except Exception as e:
            print(f"Lỗi khi so sánh với trainset: {e}")
            return None
    
    def calculate_similarity(self, features1, features2):
        """Tính toán độ tương tự giữa 2 bộ đặc trưng"""
        try:
            # Chọn các đặc trưng quan trọng để so sánh
            important_features = [
                'pitch_mean', 'pitch_std', 'spectral_centroid_mean', 'spectral_centroid_std',
                'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean',
                'zcr_mean', 'rms_mean', 'tempo', 'loudness'
            ]
            
            # Tính cosine similarity
            vec1 = np.array([features1.get(f, 0) for f in important_features])
            vec2 = np.array([features2.get(f, 0) for f in important_features])
            
            # Normalize vectors
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
            
            similarity = np.dot(vec1_norm, vec2_norm)
            return float(similarity)
            
        except Exception as e:
            print(f"Lỗi khi tính similarity: {e}")
            return 0.0
    
    def find_similar_voices(self, audio_path):
        """Tìm giọng tương tự cho file audio đã upload"""
        print(f"\n=== Tìm giọng tương tự cho: {os.path.basename(audio_path)} ===")
        
        # Upload và trích xuất đặc trưng
        result = self.upload_audio_file(audio_path)
        if not result:
            return None
        
        dest_path, features = result
        
        # So sánh với trainset
        similarities = self.compare_with_trainset(features)
        if not similarities:
            return None
        
        # Hiển thị kết quả
        print(f"\nTop 10 giọng tương tự nhất:")
        print("-" * 80)
        for i, sim in enumerate(similarities, 1):
            print(f"{i:2d}. {sim['audio_name']:<25} | Speaker: {sim['speaker']:<10} | "
                  f"Dialect: {sim['dialect']:<8} | Similarity: {sim['similarity']:.4f}")
        
        return similarities

def main():
    app = VoiceComparisonApp()
    
    print("=== Ứng dụng So sánh Giọng nói ===\n")
    print("1. Upload file audio để tìm giọng tương tự")
    print("2. So sánh với kho trainset")
    print("3. Hiển thị kết quả theo độ tương tự\n")
    
    # Ví dụ sử dụng
    audio_file = input("Nhập đường dẫn file audio cần so sánh: ").strip()
    
    if audio_file and os.path.exists(audio_file):
        app.find_similar_voices(audio_file)
    else:
        print("File không tồn tại hoặc không hợp lệ!")

if __name__ == "__main__":
    main()
