# BÁO CÁO ĐỒ ÁN TỐT NGHIỆP
## HỆ THỐNG TRA CỨU GIỌNG NÓI DỰA TRÊN ĐẶC TRƯNG ÂM THANH

---

## MỤC LỤC

**LỜI CẢM ƠN** .................................................. 1

**MỞ ĐẦU** ...................................................... 2

**MỤC LỤC** ..................................................... 3

**DANH MỤC TỪ VIẾT TẮT** ......................................... 4

**DANH MỤC HÌNH VẼ** ............................................. 5

**DANH MỤC BẢNG BIỂU** ........................................... 6

---

## **CHƯƠNG 1: TỔNG QUAN VỀ TRA CỨU GIỌNG NÓI** ................. 7

### 1.1. Tổng quan bài toán tra cứu giọng nói ................... 7
#### 1.1.1. Khái niệm tra cứu giọng nói ........................ 7
#### 1.1.2. Các phương pháp tra cứu giọng nói .................. 8
#### 1.1.3. Ứng dụng thực tế của tra cứu giọng nói .............. 9

### 1.2. Tra cứu giọng nói dựa trên đặc trưng âm thanh ........... 10
#### 1.2.1. Đặc trưng âm thanh cơ bản ........................... 10
#### 1.2.2. Đặc trưng âm thanh nâng cao ........................ 11
#### 1.2.3. So sánh các phương pháp trích xuất đặc trưng ....... 12

### 1.3. Thách thức trong tra cứu giọng nói ..................... 13
#### 1.3.1. Đa dạng về vùng miền và giọng nói .................. 13
#### 1.3.2. Nhiễu âm thanh và chất lượng âm thanh .............. 14
#### 1.3.3. Độ dài và định dạng file âm thanh .................. 15

### 1.4. Các nghiên cứu liên quan ............................... 16
#### 1.4.1. Nghiên cứu trong nước ............................... 16
#### 1.4.2. Nghiên cứu quốc tế .................................. 17
#### 1.4.3. So sánh và đánh giá các phương pháp hiện tại ....... 18

---

## **CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ THUẬT TOÁN** ................. 19

### 2.1. Xử lý tín hiệu âm thanh ............................... 19
#### 2.1.1. Tín hiệu âm thanh số ............................... 19
#### 2.1.2. Biến đổi Fourier và phổ tần số ..................... 20
#### 2.1.3. Các tham số cơ bản của âm thanh .................... 21

### 2.2. Trích xuất đặc trưng âm thanh ......................... 22
#### 2.2.1. Đặc trưng thời gian (Temporal Features) ............ 22
##### 2.2.1.1. Zero Crossing Rate (ZCR) ......................... 22
##### 2.2.1.2. Root Mean Square (RMS) Energy ................... 23
##### 2.2.1.3. Tempo và Rhythm ................................. 24

#### 2.2.2. Đặc trưng tần số (Spectral Features) ................ 25
##### 2.2.2.1. Spectral Centroid ............................... 25
##### 2.2.2.2. Spectral Bandwidth ............................... 26
##### 2.2.2.3. Spectral Flatness ................................ 27
##### 2.2.2.4. Spectral Slope .................................. 28

#### 2.2.3. Đặc trưng cepstral (Cepstral Features) ............. 29
##### 2.2.3.1. Mel-Frequency Cepstral Coefficients (MFCC) ....... 29
##### 2.2.3.2. Delta và Delta-Delta MFCC ....................... 30

#### 2.2.4. Đặc trưng nâng cao .................................. 31
##### 2.2.4.1. Pitch và Fundamental Frequency .................. 31
##### 2.2.4.2. Harmonic-to-Noise Ratio (HNR) ................... 32
##### 2.2.4.3. Loudness và Perceptual Features .................. 33

### 2.3. Thuật toán Machine Learning .......................... 34
#### 2.3.1. K-Nearest Neighbors (K-NN) ......................... 34
##### 2.3.1.1. Nguyên lý hoạt động ............................. 34
##### 2.3.1.2. Các metric khoảng cách ........................... 35
##### 2.3.1.3. Ưu nhược điểm ................................... 36

#### 2.3.2. Random Forest Classifier ........................... 37
##### 2.3.2.1. Nguyên lý hoạt động ............................. 37
##### 2.3.2.2. Ensemble Learning ................................ 38
##### 2.3.2.3. Feature Importance ............................... 39

#### 2.3.3. Support Vector Machine (SVM) ........................ 40
##### 2.3.3.1. Nguyên lý hoạt động ............................. 40
##### 2.3.3.2. Kernel Functions ................................. 41
##### 2.3.3.3. Multiclass Classification ........................ 42

### 2.4. Đánh giá hiệu suất hệ thống .......................... 43
#### 2.4.1. Metrics đánh giá ................................... 43
##### 2.4.1.1. Accuracy, Precision, Recall ...................... 43
##### 2.4.1.2. F1-Score và Confusion Matrix .................... 44
##### 2.4.1.3. Cross-Validation .................................. 45

#### 2.4.2. So sánh thuật toán .................................. 46
##### 2.4.2.1. Phương pháp so sánh .............................. 46
##### 2.4.2.2. Kết quả thực nghiệm ............................. 47

---

## **CHƯƠNG 3: THIẾT KẾ VÀ XÂY DỰNG HỆ THỐNG** ................. 48

### 3.1. Phân tích yêu cầu hệ thống ........................... 48
#### 3.1.1. Yêu cầu chức năng .................................. 48
#### 3.1.2. Yêu cầu phi chức năng .............................. 49
#### 3.1.3. Yêu cầu về dữ liệu .................................. 50

### 3.2. Kiến trúc hệ thống ................................... 51
#### 3.2.1. Kiến trúc tổng quan ................................ 51
#### 3.2.2. Các thành phần chính ............................... 52
#### 3.2.3. Luồng xử lý dữ liệu ................................ 53

### 3.3. Thiết kế cơ sở dữ liệu ............................... 54
#### 3.3.1. Cấu trúc dữ liệu âm thanh .......................... 54
#### 3.3.2. Metadata và đặc trưng ............................... 55
#### 3.3.3. Quan hệ giữa các bảng ............................... 56

### 3.4. Thiết kế giao diện người dùng ........................ 57
#### 3.4.1. Nguyên tắc thiết kế UI/UX ......................... 57
#### 3.4.2. Giao diện desktop application ....................... 58
#### 3.4.3. Workflow tương tác người dùng ....................... 59

### 3.5. Công nghệ và công cụ sử dụng ......................... 60
#### 3.5.1. Thư viện xử lý âm thanh ............................ 60
#### 3.5.2. Framework Machine Learning ......................... 61
#### 3.5.3. Công cụ phát triển ................................. 62

---

## **CHƯƠNG 4: TRIỂN KHAI VÀ THỰC NGHIỆM** .................... 63

### 4.1. Chuẩn bị dữ liệu ..................................... 63
#### 4.1.1. Thu thập dữ liệu âm thanh .......................... 63
#### 4.1.2. Tiền xử lý dữ liệu ................................. 64
#### 4.1.3. Phân chia tập dữ liệu ............................... 65

### 4.2. Triển khai hệ thống .................................. 66
#### 4.2.1. Module trích xuất đặc trưng ........................ 66
#### 4.2.2. Module huấn luyện mô hình .......................... 67
#### 4.2.3. Module tra cứu và so sánh ........................... 68
#### 4.2.4. Module giao diện người dùng ......................... 69

### 4.3. Thực nghiệm và đánh giá ............................. 70
#### 4.3.1. Thiết lập môi trường thực nghiệm ................... 70
#### 4.3.2. Thực nghiệm với các thuật toán khác nhau ............ 71
#### 4.3.3. Đánh giá hiệu suất theo vùng miền ................. 72
#### 4.3.4. So sánh với các phương pháp khác ................... 73

### 4.4. Phân tích kết quả ................................... 74
#### 4.4.1. Kết quả tổng quan .................................. 74
#### 4.4.2. Phân tích chi tiết từng thuật toán ................. 75
#### 4.4.3. Đánh giá ưu nhược điểm ............................. 76
#### 4.4.4. Khuyến nghị cải tiến ............................... 77

---

## **CHƯƠNG 5: KẾT QUẢ VÀ ĐÁNH GIÁ** .......................... 78

### 5.1. Kết quả thực nghiệm ................................. 78
#### 5.1.1. Độ chính xác của hệ thống .......................... 78
#### 5.1.2. Thời gian xử lý ..................................... 79
#### 5.1.3. Hiệu suất theo từng vùng miền ...................... 80

### 5.2. So sánh với các phương pháp khác ..................... 81
#### 5.2.1. So sánh thuật toán ................................. 81
#### 5.2.2. So sánh đặc trưng ................................... 82
#### 5.2.3. So sánh hiệu suất tổng thể ......................... 83

### 5.3. Đánh giá hệ thống .................................... 84
#### 5.3.1. Đánh giá chức năng .................................. 84
#### 5.3.2. Đánh giá hiệu suất .................................. 85
#### 5.3.3. Đánh giá trải nghiệm người dùng .................... 86

### 5.4. Hạn chế và thách thức ............................... 87
#### 5.4.1. Hạn chế về dữ liệu ................................. 87
#### 5.4.2. Hạn chế về thuật toán .............................. 88
#### 5.4.3. Thách thức trong thực tế ........................... 89

---

## **CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN** ................ 90

### 6.1. Kết luận ............................................. 90
#### 6.1.1. Những đóng góp chính ............................... 90
#### 6.1.2. Mục tiêu đã đạt được ................................ 91
#### 6.1.3. Ý nghĩa khoa học và thực tiễn ..................... 92

### 6.2. Hướng phát triển ..................................... 93
#### 6.2.1. Cải tiến thuật toán ................................ 93
#### 6.2.2. Mở rộng dữ liệu ..................................... 94
#### 6.2.3. Phát triển ứng dụng ................................. 95
#### 6.2.4. Tích hợp công nghệ mới ............................. 96

### 6.3. Khuyến nghị ......................................... 97
#### 6.3.1. Khuyến nghị cho nghiên cứu tiếp theo .............. 97
#### 6.3.2. Khuyến nghị cho ứng dụng thực tế ................... 98

---

## **TÀI LIỆU THAM KHẢO** ..................................... 99

## **PHỤ LỤC** ................................................ 105
### Phụ lục A: Mã nguồn chính ................................ 105
### Phụ lục B: Kết quả thực nghiệm chi tiết .................. 110
### Phụ lục C: Hướng dẫn sử dụng hệ thống .................... 115

---

## **DANH MỤC TỪ VIẾT TẮT**

| Viết tắt | Ý nghĩa |
|----------|---------|
| CBIR | Content-Based Image Retrieval |
| CBVR | Content-Based Voice Retrieval |
| MFCC | Mel-Frequency Cepstral Coefficients |
| K-NN | K-Nearest Neighbors |
| SVM | Support Vector Machine |
| RF | Random Forest |
| ZCR | Zero Crossing Rate |
| RMS | Root Mean Square |
| HNR | Harmonic-to-Noise Ratio |
| FFT | Fast Fourier Transform |
| STFT | Short-Time Fourier Transform |
| UI | User Interface |
| UX | User Experience |
| API | Application Programming Interface |
| CSV | Comma-Separated Values |
| JSON | JavaScript Object Notation |
| GUI | Graphical User Interface |

---

## **DANH MỤC HÌNH VẼ**

| Số thứ tự | Tên hình | Trang |
|-----------|----------|-------|
| Hình 1.1 | So sánh tra cứu ảnh và tra cứu giọng nói | 7 |
| Hình 1.2 | Các phương pháp tra cứu giọng nói | 8 |
| Hình 1.3 | Ứng dụng thực tế của tra cứu giọng nói | 9 |
| Hình 2.1 | Cấu trúc tín hiệu âm thanh số | 19 |
| Hình 2.2 | Biến đổi Fourier và phổ tần số | 20 |
| Hình 2.3 | Đặc trưng thời gian (ZCR, RMS) | 22 |
| Hình 2.4 | Đặc trưng tần số (Spectral Centroid) | 25 |
| Hình 2.5 | Quá trình trích xuất MFCC | 29 |
| Hình 2.6 | Nguyên lý hoạt động K-NN | 34 |
| Hình 2.7 | Các metric khoảng cách | 35 |
| Hình 2.8 | Nguyên lý Random Forest | 37 |
| Hình 2.9 | SVM với các kernel khác nhau | 40 |
| Hình 3.1 | Kiến trúc tổng quan hệ thống | 51 |
| Hình 3.2 | Các thành phần chính | 52 |
| Hình 3.3 | Luồng xử lý dữ liệu | 53 |
| Hình 3.4 | Cấu trúc cơ sở dữ liệu | 54 |
| Hình 3.5 | Giao diện desktop application | 58 |
| Hình 3.6 | Workflow tương tác người dùng | 59 |
| Hình 4.1 | Quy trình chuẩn bị dữ liệu | 63 |
| Hình 4.2 | Module trích xuất đặc trưng | 66 |
| Hình 4.3 | Module huấn luyện mô hình | 67 |
| Hình 4.4 | Module tra cứu và so sánh | 68 |
| Hình 4.5 | Giao diện người dùng | 69 |
| Hình 5.1 | Biểu đồ độ chính xác | 78 |
| Hình 5.2 | Biểu đồ thời gian xử lý | 79 |
| Hình 5.3 | So sánh hiệu suất theo vùng miền | 80 |
| Hình 5.4 | So sánh các thuật toán | 81 |
| Hình 5.5 | Confusion Matrix | 84 |

---

## **DANH MỤC BẢNG BIỂU**

| Số thứ tự | Tên bảng | Trang |
|-----------|----------|-------|
| Bảng 1.1 | So sánh các phương pháp tra cứu | 8 |
| Bảng 2.1 | Các đặc trưng âm thanh được sử dụng | 22 |
| Bảng 2.2 | So sánh các metric khoảng cách | 35 |
| Bảng 2.3 | So sánh các thuật toán ML | 46 |
| Bảng 3.1 | Yêu cầu chức năng hệ thống | 48 |
| Bảng 3.2 | Yêu cầu phi chức năng | 49 |
| Bảng 3.3 | Cấu trúc metadata | 55 |
| Bảng 4.1 | Thống kê dữ liệu âm thanh | 63 |
| Bảng 4.2 | Phân chia tập dữ liệu | 65 |
| Bảng 4.3 | Cấu hình môi trường thực nghiệm | 70 |
| Bảng 5.1 | Kết quả độ chính xác | 78 |
| Bảng 5.2 | Thời gian xử lý | 79 |
| Bảng 5.3 | Hiệu suất theo vùng miền | 80 |
| Bảng 5.4 | So sánh thuật toán | 81 |
| Bảng 5.5 | Đánh giá hệ thống | 84 |

---

## **GỢI Ý NỘI DUNG CHI TIẾT CHO TỪNG CHƯƠNG**

### **CHƯƠNG 1: TỔNG QUAN**
- Giới thiệu bài toán tra cứu giọng nói
- So sánh với tra cứu ảnh (CBIR)
- Các ứng dụng thực tế: nhận dạng người nói, phân loại giọng, tìm kiếm audio
- Thách thức: đa dạng vùng miền, nhiễu, chất lượng âm thanh

### **CHƯƠNG 2: CƠ SỞ LÝ THUYẾT**
- Xử lý tín hiệu âm thanh: FFT, STFT
- 15+ đặc trưng âm thanh: Pitch, MFCC, Spectral, Temporal
- Thuật toán ML: K-NN, RandomForest, SVM
- So sánh ưu nhược điểm từng thuật toán

### **CHƯƠNG 3: THIẾT KẾ HỆ THỐNG**
- Kiến trúc 3 tầng: Training, Inference, Testing
- Module hóa: Feature Extraction, Model Training, Voice Comparison
- Giao diện desktop với tkinter
- Workflow: Upload → Extract → Compare → Display

### **CHƯƠNG 4: TRIỂN KHAI**
- Dữ liệu: 8,166 files MP3 từ ViSpeech dataset
- Implementation: Python + Librosa + Scikit-learn
- Testing: 12 thuật toán ML khác nhau
- Evaluation: Accuracy, Time, Regional performance

### **CHƯƠNG 5: KẾT QUẢ**
- Accuracy: 85-90% (top-5 matches)
- Regional detection: 80-85%
- Processing time: <100ms per query
- So sánh với các phương pháp khác

### **CHƯƠNG 6: KẾT LUẬN**
- Đóng góp: Hệ thống tra cứu giọng nói đa vùng miền
- Hướng phát triển: Deep Learning, Real-time processing
- Ứng dụng: Forensic, Security, Entertainment
