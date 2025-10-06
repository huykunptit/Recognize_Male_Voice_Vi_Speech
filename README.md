# ViSpeech - Hệ thống So sánh Giọng nói

Hệ thống phân tích và so sánh giọng nói với metadata mở rộng và khả năng tìm kiếm giọng tương tự.

## 🚀 Tính năng chính

### 1. Lọc file audio theo giới tính Male
- Script: `filter_male_audio.py`
- Chức năng: Lọc và chỉ giữ lại các file audio có gender = "Male"
- Backup an toàn trước khi xóa

### 2. Database Speaker với tên đàn ông Việt Nam
- Script: `create_speaker_database.py`
- Output: `speaker_database.csv`
- Chức năng: Tạo database mapping speaker ID với tên đàn ông Việt Nam

### 3. So sánh giọng nói
- Script: `voice_comparison_app.py`
- Chức năng: Upload file audio và tìm giọng tương tự nhất
- Sử dụng 15+ đặc trưng âm thanh để so sánh

### 4. Super Metadata với 15+ trường thông tin
- Script: `create_super_metadata.py`
- Output: Folder `super_metadata/` với 3 file CSV mở rộng
- Chức năng: Mở rộng metadata với đặc trưng âm thanh chi tiết

## 📁 Cấu trúc thư mục

```
ViSpeech/
├── metadata/                    # Metadata gốc
│   ├── clean_testset.csv
│   ├── noisy_testset.csv
│   └── trainset.csv
├── super_metadata/              # Metadata mở rộng
│   ├── clean_testset.csv
│   ├── noisy_testset.csv
│   └── trainset.csv
├── data_check/                  # Folder upload audio
├── backup_deleted_files/        # Backup files đã xóa
├── speaker_database.csv         # Database speaker
└── scripts/
    ├── filter_male_audio.py
    ├── create_speaker_database.py
    ├── voice_comparison_app.py
    ├── create_super_metadata.py
    └── run_all_scripts.py
```

## 🛠️ Cài đặt

1. Cài đặt Python dependencies:
```bash
pip install -r requirements.txt
```

2. Cài đặt thêm librosa (nếu cần):
```bash
pip install librosa
```

## 📖 Hướng dẫn sử dụng

### 1. Chạy tất cả scripts
```bash
python run_all_scripts.py
```

### 2. Lọc file audio theo giới tính Male
```bash
python filter_male_audio.py
```

### 3. Tạo database speaker
```bash
python create_speaker_database.py
```

### 4. So sánh giọng nói
```bash
python voice_comparison_app.py
```

### 5. Tạo super metadata
```bash
python create_super_metadata.py
```

### 6. Train đặc trưng âm thanh thực tế
```bash
python run_training.py
```

## 🎵 15+ Đặc trưng âm thanh được trích xuất

1. **Pitch (Độ cao giọng)**: Mean, Std, Range
2. **Spectral Centroid (Độ trầm bổng)**: Mean, Std
3. **Spectral Rolloff (Độ rõ ràng)**: Mean, Std
4. **Zero Crossing Rate**: Mean, Std
5. **MFCC (13 hệ số)**: Mean, Std cho mỗi hệ số
6. **Chroma Features**: Mean, Std
7. **Spectral Contrast**: Mean, Std
8. **Tonnetz**: Mean, Std
9. **RMS Energy**: Mean, Std, Max, Min
10. **Tempo**: Nhịp độ
11. **Duration**: Thời lượng
12. **Loudness**: Độ to (dB)
13. **Spectral Bandwidth**: Mean, Std
14. **Spectral Flatness**: Mean, Std
15. **Harmonic-to-Noise Ratio**: Tỷ lệ hài hòa/nhiễu
16. **Spectral Slope**: Mean, Std
17. **Spectral Kurtosis**: Mean, Std
18. **Spectral Skewness**: Mean, Std
19. **Onset Strength**: Mean, Std
20. **Spectral Flux**: Dòng phổ

## 🔍 So sánh giọng nói

Hệ thống sử dụng thuật toán cosine similarity để so sánh các đặc trưng âm thanh:

1. Upload file audio cần so sánh
2. Trích xuất 15+ đặc trưng âm thanh
3. So sánh với kho trainset
4. Hiển thị top 10 giọng tương tự nhất

## 📊 Kết quả

- **Speaker Database**: Mapping speaker ID với tên đàn ông Việt Nam
- **Super Metadata**: 3 file CSV với 20+ trường thông tin âm thanh
- **Voice Comparison**: Tìm kiếm giọng tương tự với độ chính xác cao
- **Audio Filtering**: Lọc file audio theo giới tính Male

## ⚠️ Lưu ý

- Đảm bảo có đủ dung lượng ổ cứng cho việc xử lý audio
- Quá trình trích xuất đặc trưng có thể mất thời gian với dataset lớn
- Backup files trước khi chạy script lọc audio
- Cần cài đặt đầy đủ dependencies trước khi chạy

## 🐛 Troubleshooting

1. **Lỗi librosa**: Cài đặt thêm `pip install librosa`
2. **Lỗi soundfile**: Cài đặt thêm `pip install soundfile`
3. **Lỗi memory**: Giảm batch size hoặc xử lý từng file nhỏ
4. **Lỗi encoding**: Đảm bảo file CSV có encoding UTF-8
