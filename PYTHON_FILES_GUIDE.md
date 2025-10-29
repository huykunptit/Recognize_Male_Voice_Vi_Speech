# ViSpeech - Voice Comparison System

## 📁 Cấu trúc file Python (Đã lọc)

### 🚀 **File chính để chạy ứng dụng:**

1. **`run_final_app.py`** - **MAIN ENTRY POINT**
   - Khởi động ứng dụng chính
   - Cho phép chọn giữa 2 loại app:
     - Desktop App Final (cơ bản)
     - Auto Regional Detection App (tự động phát hiện vùng miền)

2. **`run_auto_regional_app.py`** - Auto Regional Detection App
   - Khởi động app với tính năng tự động phát hiện vùng miền
   - So sánh chính xác hơn với lọc theo vùng miền

3. **`run_regional_app.py`** - Regional App
   - Khởi động app với tính năng chọn vùng miền thủ công
   - Cho phép checkbox chọn North/Central/South

### 🛠️ **File training và xử lý dữ liệu:**

4. **`run_training.py`** - Training chính
   - Chạy training dữ liệu cho toàn bộ dataset
   - Tạo super metadata với 15+ thuộc tính audio

5. **`run_regional_training.py`** - Regional Training
   - Training dữ liệu theo vùng miền
   - Cho phép chọn vùng miền để training

6. **`train_audio_features.py`** - Core Training Engine
   - Engine chính để trích xuất thuộc tính audio
   - Xử lý file MP3 và tạo metadata

7. **`train_regional.py`** - Regional Training Engine
   - Engine training theo vùng miền
   - Lọc dữ liệu theo North/Central/South

### 🎯 **File ứng dụng desktop:**

8. **`voice_desktop_app_final.py`** - Desktop App Final
   - Ứng dụng desktop cơ bản
   - Có đầy đủ tính năng: upload, ghi âm, replay, pause
   - Hiển thị thuộc tính audio và kết quả so sánh

9. **`voice_auto_regional_app.py`** - Auto Regional Detection App
   - Ứng dụng desktop với tự động phát hiện vùng miền
   - Sử dụng RandomForestClassifier
   - So sánh có lọc theo vùng miền phát hiện

10. **`voice_regional_app.py`** - Regional App
    - Ứng dụng desktop với chọn vùng miền thủ công
    - Checkbox để chọn vùng miền
    - Training và so sánh theo vùng miền đã chọn

### 🔧 **File tiện ích:**

11. **`create_speaker_database.py`** - Tạo Speaker Database
    - Tạo file `speaker_database.csv`
    - Map speaker ID với tên tiếng Việt

12. **`create_super_metadata.py`** - Tạo Super Metadata
    - Tạo file metadata với 15+ thuộc tính audio
    - Xử lý encoding cho Windows

13. **`merge_male_only.py`** - Merge Male Only Dataset
    - Gộp các file CSV từ `super_metadata/male_only/`
    - Tạo file `male_only_merged.csv`

## 🚀 **Cách sử dụng:**

### **Bắt đầu nhanh:**
```bash
python run_final_app.py
```

### **Training dữ liệu:**
```bash
python run_training.py
```

### **Training theo vùng miền:**
```bash
python run_regional_training.py
```

### **App tự động phát hiện vùng miền:**
```bash
python run_auto_regional_app.py
```

## 📋 **Workflow đề xuất:**

1. **Setup ban đầu:**
   ```bash
   python create_speaker_database.py
   python run_training.py
   ```

2. **Sử dụng app chính:**
   ```bash
   python run_final_app.py
   # Chọn option 2: Auto Regional Detection App
   ```

3. **Training theo vùng miền (tùy chọn):**
   ```bash
   python run_regional_training.py
   ```

## 🎯 **Tính năng chính:**

- ✅ Upload/Ghi âm audio (tự động cắt 20s)
- ✅ Trích xuất 15+ thuộc tính audio
- ✅ So sánh giọng nói với K-NN
- ✅ Replay/Pause audio
- ✅ Hiển thị thuộc tính JSON
- ✅ Tự động phát hiện vùng miền
- ✅ So sánh có lọc theo vùng miền
- ✅ Training theo vùng miền

## 📊 **Dữ liệu:**

- **Trainset**: 8,166 files MP3
- **Vùng miền**: North (2,814), Central (2,472), South (2,880)
- **Thuộc tính**: 15+ đặc trưng audio (Pitch, MFCC, Spectral, etc.)
- **Output**: JSON với thuộc tính + kết quả phát hiện vùng miền
