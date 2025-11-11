# CHƯƠNG 2: XÂY DỰNG HỆ THỐNG

## 2.1. Thiết kế hệ thống

### 2.1.1. Sơ đồ khối của hệ thống:

![Sơ đồ khối hệ thống](flowchart.html)

**Hình 2.1. Sơ đồ khối hệ thống**

Sơ đồ khối trên mô tả hệ thống tìm kiếm giọng nói tương tự với các thành phần chính:

- **Khởi tạo hệ thống**: Khởi tạo giao diện người dùng (GUI) và engine xử lý giọng nói
- **Quản lý Model**: Kiểm tra và load model đã train, hoặc train model mới từ dữ liệu training
- **Thu thập Input**: Người dùng chọn file audio hoặc ghi âm trực tiếp
- **Trích xuất đặc trưng**: Trích xuất các đặc trưng audio từ file input
- **Chuẩn hóa dữ liệu**: Chuẩn hóa feature vector để đảm bảo tính nhất quán
- **Tìm kiếm KNN**: Sử dụng thuật toán K-Nearest Neighbors để tìm các giọng nói tương tự
- **Lọc và sắp xếp**: Lọc theo vùng miền (nếu cần) và sắp xếp kết quả theo độ tương tự
- **Hiển thị kết quả**: Hiển thị danh sách các giọng nói tương tự trong giao diện

### 2.1.2. Kiến trúc hệ thống

Hệ thống được xây dựng dựa trên kiến trúc client-server với các thành phần:

1. **VoiceSearchEngine**: Lớp xử lý chính, chịu trách nhiệm:
   - Load và quản lý dữ liệu training
   - Trích xuất đặc trưng audio
   - Huấn luyện và sử dụng mô hình KNN
   - Tìm kiếm giọng nói tương tự

2. **VoiceSearchApp**: Lớp giao diện người dùng (GUI), chịu trách nhiệm:
   - Hiển thị giao diện tương tác
   - Xử lý các sự kiện người dùng
   - Hiển thị kết quả tìm kiếm
   - Quản lý phát/ghi âm audio

3. **Dữ liệu Training**: 
   - Thư mục `super_metadata/male_only/` chứa các file CSV metadata
   - File `speaker_database.csv` chứa thông tin người nói
   - Model đã train: `trained_model.json`, `scaler.joblib`, `knn_model.joblib`

## 2.2. Thu thập và chuẩn hóa dữ liệu

### 2.2.1. Thu thập dữ liệu

Hệ thống sử dụng dữ liệu audio được thu thập từ nhiều nguồn khác nhau:

- **Dữ liệu training**: Tập hợp các file audio từ thư mục `trainset/` với 8166 file audio
- **Metadata**: Các file CSV trong thư mục `super_metadata/male_only/` chứa thông tin đặc trưng đã được trích xuất
- **Speaker Database**: File `speaker_database.csv` chứa thông tin người nói bao gồm:
  - Speaker ID
  - Tên tiếng Việt
  - Vùng miền (Bắc, Trung, Nam)
  - Giới tính

### 2.2.2. Chuẩn hóa dữ liệu

Quá trình chuẩn hóa dữ liệu được thực hiện qua các bước:

1. **Load dữ liệu training**: 
   - Đọc tất cả các file CSV từ thư mục `super_metadata/male_only/`
   - Merge các DataFrame thành một tập dữ liệu duy nhất
   - Loại bỏ các cột không phải feature (audio_name, dialect, gender, speaker)

2. **Xử lý dữ liệu thiếu**:
   - Thay thế các giá trị NaN bằng 0
   - Đảm bảo tất cả các feature có giá trị hợp lệ

3. **Chuẩn hóa bằng StandardScaler**:
   - Sử dụng `StandardScaler` từ thư viện scikit-learn
   - Chuẩn hóa dữ liệu về phân phối chuẩn với mean=0 và std=1
   - Công thức: `X_scaled = (X - mean) / std`
   - Lưu scaler để sử dụng cho dữ liệu mới

4. **Lưu trữ model**:
   - Lưu scaler vào file `scaler.joblib`
   - Lưu KNN model vào file `knn_model.joblib`
   - Lưu thông tin training vào file `trained_model.json`

### 2.2.3. Cấu trúc dữ liệu

Mỗi mẫu audio trong dữ liệu training được biểu diễn bằng một vector đặc trưng gồm 50+ features:

- **Pitch features**: pitch_mean, pitch_std, pitch_range
- **MFCC features**: 13 MFCC coefficients (mean và std cho mỗi coefficient)
- **Spectral features**: spectral_centroid, spectral_rolloff, spectral_bandwidth, spectral_flatness, spectral_contrast
- **Temporal features**: ZCR (Zero Crossing Rate), RMS, tempo, duration
- **Advanced features**: chroma, tonnetz, spectral_slope, spectral_kurtosis, spectral_skewness, spectral_flux, onset_strength
- **Metadata**: audio_name, dialect, gender, speaker

## 2.3. Trích rút đặc trưng

### 2.3.1. Tổng quan về trích rút đặc trưng

Hệ thống trích xuất các đặc trưng audio từ file input để tạo ra một vector đặc trưng có thể so sánh với các mẫu trong database. Quá trình trích xuất được thực hiện bằng thư viện librosa và các công cụ xử lý tín hiệu số.

### 2.3.2. Các loại đặc trưng được trích xuất

#### 2.3.2.1. Đặc trưng Pitch (Tần số cơ bản)

- **pitch_mean**: Giá trị trung bình của pitch (Hz)
- **pitch_std**: Độ lệch chuẩn của pitch
- **pitch_range**: Khoảng giá trị pitch (max - min)

**Phương pháp**: Sử dụng thuật toán PYIN (Probabilistic YIN) hoặc YIN làm phương án dự phòng để ước lượng pitch từ tín hiệu audio.

#### 2.3.2.2. Đặc trưng MFCC (Mel-Frequency Cepstral Coefficients)

MFCC là đặc trưng quan trọng nhất trong nhận dạng giọng nói, mô phỏng cách xử lý âm thanh của tai người.

- Trích xuất 13 MFCC coefficients
- Tính mean và std cho mỗi coefficient
- Tổng cộng: 26 features (mfcc_1_mean đến mfcc_13_std)

**Công thức**: MFCC được tính từ phổ Mel, sau đó áp dụng DCT (Discrete Cosine Transform).

#### 2.3.2.3. Đặc trưng Spectral (Phổ tần số)

- **Spectral Centroid**: Trọng tâm phổ tần số, đại diện cho "độ sáng" của âm thanh
- **Spectral Rolloff**: Tần số dưới đó chứa 85% năng lượng phổ
- **Spectral Bandwidth**: Độ rộng phổ tần số
- **Spectral Flatness**: Độ phẳng của phổ, đo tính "noise-like" của tín hiệu
- **Spectral Contrast**: Độ tương phản giữa các peak và valley trong phổ
- **Spectral Slope**: Độ dốc của phổ log-magnitude
- **Spectral Kurtosis**: Độ nhọn của phân phối phổ
- **Spectral Skewness**: Độ lệch của phân phối phổ
- **Spectral Flux**: Tốc độ thay đổi của phổ theo thời gian

#### 2.3.2.4. Đặc trưng Temporal (Thời gian)

- **ZCR (Zero Crossing Rate)**: Tần suất tín hiệu đổi dấu, đặc trưng cho tính chất âm thanh
- **RMS (Root Mean Square)**: Năng lượng trung bình của tín hiệu
- **Tempo**: Nhịp độ của audio (BPM)
- **Duration**: Độ dài của file audio (giây)
- **Onset Strength**: Độ mạnh của các điểm bắt đầu âm thanh

#### 2.3.2.5. Đặc trưng Harmonic (Hài hòa)

- **Chroma**: Đặc trưng biểu diễn thông tin về cao độ âm nhạc (12 nốt nhạc)
- **Tonnetz**: Biểu diễn mối quan hệ hài hòa giữa các nốt nhạc
- **HNR (Harmonic-to-Noise Ratio)**: Tỷ lệ tín hiệu hài hòa so với nhiễu

#### 2.3.2.6. Đặc trưng Loudness (Độ lớn)

- **Loudness**: Độ lớn của âm thanh (dB), tính từ RMS
- **Loudness Peak**: Độ lớn đỉnh

**Công thức**: `Loudness = 20 * log10(RMS + epsilon)`

### 2.3.3. Quy trình trích xuất đặc trưng

1. **Load audio file**:
   - Sử dụng `soundfile` hoặc `librosa` để đọc file
   - Chuyển đổi sang mono nếu là stereo
   - Normalize tín hiệu về dải [-1, 1]

2. **Xử lý tín hiệu**:
   - STFT (Short-Time Fourier Transform) với n_fft=2048, hop_length=512
   - Tính toán các đặc trưng từ phổ tần số và tín hiệu thời gian

3. **Tính toán thống kê**:
   - Với mỗi đặc trưng, tính mean và std qua các frame
   - Một số đặc trưng tính thêm min, max, range

4. **Xử lý lỗi**:
   - Nếu không thể trích xuất một đặc trưng, sử dụng giá trị mặc định (0.0)
   - Đảm bảo vector đặc trưng luôn có đầy đủ các thành phần

5. **Trả về vector đặc trưng**:
   - Dictionary chứa tất cả các features
   - Đảm bảo thứ tự và số lượng features giống với dữ liệu training

## 2.4. Tìm kiếm giọng nói tương tự

### 2.4.1. Thuật toán K-Nearest Neighbors (KNN)

Hệ thống sử dụng thuật toán KNN để tìm K giọng nói tương tự nhất trong database. KNN là một thuật toán học máy không tham số, hoạt động dựa trên nguyên lý "những mẫu gần nhau thường có đặc điểm tương tự".

**Các bước thực hiện**:

1. **Chuẩn bị dữ liệu**:
   - Trích xuất đặc trưng từ file audio input
   - Chuẩn hóa feature vector bằng scaler đã train
   - Đảm bảo vector có cùng số chiều với dữ liệu training

2. **Tìm K neighbors**:
   - Sử dụng `NearestNeighbors` từ scikit-learn
   - Tìm K mẫu gần nhất trong không gian đặc trưng
   - K mặc định = 10, có thể điều chỉnh

3. **Tính độ tương tự**:
   - Chuyển đổi distance thành similarity score
   - Công thức: `similarity = (1 - distance) * 100`

4. **Lọc và sắp xếp**:
   - Lọc theo vùng miền nếu được yêu cầu
   - Boost similarity cho cùng vùng miền (+20%)
   - Sắp xếp theo similarity giảm dần

### 2.4.2. Các phương pháp đo khoảng cách

Hệ thống hỗ trợ nhiều phương pháp đo khoảng cách khác nhau, mặc định sử dụng Cosine Similarity.

#### 2.4.2.1. Cosine Similarity (Khoảng cách Cosine)

Cosine Similarity đo độ tương tự dựa trên góc giữa hai vector trong không gian đặc trưng.

**Công thức**:

```
cosine_similarity = (A · B) / (||A|| × ||B||)
```

**Ưu điểm**:
- Không phụ thuộc vào độ lớn của vector, chỉ phụ thuộc vào hướng
- Phù hợp với dữ liệu đã được chuẩn hóa
- Hiệu quả với dữ liệu thưa (sparse data)

**Nhược điểm**:
- Không tính đến độ lớn của vector
- Có thể không phù hợp với một số loại đặc trưng

**Trong hệ thống**: Cosine Similarity là metric mặc định được sử dụng trong KNN model.

#### 2.4.2.2. Euclidean Distance (Khoảng cách Euclid)

Euclidean Distance đo khoảng cách thẳng giữa hai điểm trong không gian đặc trưng.

**Công thức**:

```
euclidean_distance = √(Σ(Ai - Bi)²)
```

**Ưu điểm**:
- Dễ hiểu và tính toán
- Phù hợp với dữ liệu liên tục
- Hiệu quả với dữ liệu có số chiều thấp

**Nhược điểm**:
- Bị ảnh hưởng bởi các feature có giá trị lớn
- Cần chuẩn hóa dữ liệu trước khi sử dụng
- Hiệu suất giảm với dữ liệu có số chiều cao

**Ứng dụng**: Có thể sử dụng trong KNN bằng cách thay đổi metric từ 'cosine' sang 'euclidean'.

#### 2.4.2.3. Manhattan Distance (Khoảng cách Manhattan)

Manhattan Distance (còn gọi là L1 distance) đo tổng khoảng cách theo các trục tọa độ.

**Công thức**:

```
manhattan_distance = Σ|Ai - Bi|
```

**Ưu điểm**:
- Ít nhạy cảm với outliers hơn Euclidean
- Phù hợp với dữ liệu có nhiều giá trị 0 (sparse data)
- Tính toán nhanh

**Nhược điểm**:
- Không phù hợp với dữ liệu có số chiều cao
- Có thể không phản ánh chính xác mối quan hệ trong không gian đặc trưng

**Ứng dụng**: Có thể sử dụng trong KNN với metric='manhattan'.

#### 2.4.2.4. Correlation Distance (Khoảng cách tương quan)

Correlation Distance đo độ tương quan giữa hai vector, dựa trên hệ số tương quan Pearson.

**Công thức**:

```
correlation_distance = 1 - correlation_coefficient
```

Trong đó correlation coefficient được tính:

```
correlation = Σ((Ai - Ā)(Bi - B̄)) / √(Σ(Ai - Ā)² × Σ(Bi - B̄)²)
```

**Ưu điểm**:
- Đo độ tương quan tuyến tính giữa các vector
- Không bị ảnh hưởng bởi scale của dữ liệu
- Phù hợp với dữ liệu có mối quan hệ tuyến tính

**Nhược điểm**:
- Chỉ phát hiện mối quan hệ tuyến tính
- Tính toán phức tạp hơn các phương pháp khác
- Có thể không phù hợp với dữ liệu phi tuyến

**Ứng dụng**: Có thể sử dụng trong KNN với metric='correlation'.

### 2.4.3. So sánh các phương pháp

| Phương pháp | Metric trong KNN | Ưu điểm | Nhược điểm | Phù hợp với |
|------------|------------------|---------|------------|-------------|
| Cosine Similarity | 'cosine' | Không phụ thuộc độ lớn, hiệu quả với sparse data | Bỏ qua độ lớn vector | Dữ liệu đã chuẩn hóa, text/audio features |
| Euclidean Distance | 'euclidean' | Dễ hiểu, phù hợp dữ liệu liên tục | Bị ảnh hưởng bởi scale, chậm với high-dim | Dữ liệu số chiều thấp, đã chuẩn hóa |
| Manhattan Distance | 'manhattan' | Ít nhạy outliers, tính nhanh | Không phù hợp high-dim | Sparse data, dữ liệu có nhiều giá trị 0 |
| Correlation Distance | 'correlation' | Đo tương quan tuyến tính, không phụ thuộc scale | Chỉ phát hiện quan hệ tuyến tính | Dữ liệu có quan hệ tuyến tính |

### 2.4.4. Lọc và tối ưu kết quả

#### 2.4.4.1. Lọc theo vùng miền

Hệ thống cho phép lọc kết quả theo vùng miền (Bắc, Trung, Nam) để tìm giọng nói tương tự trong cùng một vùng miền.

**Cơ chế**:
- Người dùng chọn vùng miền từ dropdown
- Hệ thống chỉ trả về các mẫu có cùng vùng miền
- Tìm nhiều neighbors hơn (k*3) để đảm bảo đủ kết quả sau khi lọc

#### 2.4.4.2. Boost cùng vùng miền

Khi bật tính năng "Ưu tiên cùng vùng miền", hệ thống sẽ tăng similarity score thêm 20% cho các mẫu có cùng vùng miền với input.

**Công thức**:
```
similarity_boosted = min(100, base_similarity + 20)
```

**Mục đích**: Tăng độ ưu tiên cho các giọng nói cùng vùng miền, phù hợp với ứng dụng thực tế.

#### 2.4.4.3. Sắp xếp và hiển thị

- Sắp xếp kết quả theo similarity giảm dần
- Hiển thị top K kết quả với thông tin:
  - Rank (thứ hạng)
  - Similarity score (%)
  - Tên người nói (tiếng Việt)
  - Tên file audio
  - Vùng miền
  - Link để phát audio

## 2.5. Giao diện người dùng

### 2.5.1. Thiết kế giao diện

Hệ thống sử dụng Tkinter để xây dựng giao diện đồ họa (GUI) với các thành phần:

- **Frame Model Info**: Hiển thị thông tin model đã train
- **Frame File Selection**: Chọn file audio hoặc ghi âm
- **Frame Settings**: Cài đặt số kết quả, lọc vùng miền
- **Frame Results**: Hiển thị kết quả tìm kiếm dạng bảng
- **Frame Features**: Hiển thị và so sánh đặc trưng (debug mode)

### 2.5.2. Chức năng chính

1. **Train Model**: Huấn luyện model mới từ dữ liệu training
2. **Load Model**: Load model đã train từ file
3. **Chọn File**: Chọn file audio từ máy tính
4. **Ghi âm**: Ghi âm trực tiếp từ microphone
5. **Phát Audio**: Phát file audio input hoặc kết quả
6. **Tìm kiếm**: Thực hiện tìm kiếm giọng nói tương tự
7. **Xem Features**: Xem và so sánh đặc trưng (debug mode)

### 2.5.3. Xử lý đa luồng

Để tránh đóng băng giao diện khi xử lý, hệ thống sử dụng threading:

- Tìm kiếm được thực hiện trong thread riêng
- Progress bar và status label được cập nhật real-time
- Giao diện vẫn responsive trong quá trình xử lý

---

**Tóm tắt chương 2**: Chương này trình bày chi tiết về thiết kế và xây dựng hệ thống tìm kiếm giọng nói tương tự, bao gồm sơ đồ khối, quy trình thu thập và chuẩn hóa dữ liệu, các phương pháp trích xuất đặc trưng audio, và các thuật toán tìm kiếm sử dụng KNN với nhiều phương pháp đo khoảng cách khác nhau.

