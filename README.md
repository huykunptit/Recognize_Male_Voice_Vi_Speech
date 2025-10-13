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

## 📏 Khoảng giá trị tham khảo cho các đặc trưng (Low / Medium / High)

Bảng sau cung cấp các khoảng tham khảo tổng quát — giá trị thực tế phụ thuộc pipeline (sr, window, normalization). Dùng để tham khảo và so sánh tương đối trong cùng dataset.

| Feature | Unit | Low | Medium | High | Ghi chú ngắn |
|---|---:|---:|---:|---:|---|
| pitch_mean | Hz | < 100 | 100 – 220 | > 220 | Giới tính/tuổi ảnh hưởng mạnh |
| pitch_std | Hz | < 10 | 10 – 40 | > 40 | Độ biến thiên cao → biểu cảm |
| pitch_range | Hz | < 50 | 50 – 150 | > 150 | Phạm vi F0 |
| spectral_centroid_mean | Hz | < 1500 | 1500 – 3000 | > 3000 | “Ấm” ↔ “Sáng” |
| spectral_centroid_std | Hz | < 200 | 200 – 800 | > 800 | Biến đổi màu âm |
| spectral_rolloff_mean | Hz | < 2000 | 2000 – 4000 | > 4000 | Năng lượng cao tần |
| spectral_rolloff_std | Hz | < 300 | 300 – 1000 | > 1000 | --- |
| zcr_mean | ratio | < 0.01 | 0.01 – 0.1 | > 0.1 | Tiếng ồn/frasal tăng ZCR |
| zcr_std | ratio | < 0.01 | 0.01 – 0.05 | > 0.05 | --- |
| MFCC_n_mean (typical) | coeff | | | | MFCC không có đơn vị cố định; tham khảo theo abs(magnitude): low <50, med 50–150, high >150 |
| MFCC_n_std | coeff | <5 | 5 – 30 | >30 | --- |
| chroma_mean | 0–1 | <0.2 | 0.2 – 0.6 | >0.6 | Nếu normalized |
| chroma_std | 0–1 | <0.05 | 0.05 – 0.2 | >0.2 | --- |
| spectral_contrast_mean | dB | <10 | 10 – 30 | >30 | Độ khác biệt đỉnh/rãnh |
| spectral_contrast_std | dB | <2 | 2 – 8 | >8 | --- |
| tonnetz_mean | unitless | ~ -0.3..0.3 | ~ -0.3..0.3 | >|0.3| | Phụ thuộc nội dung nhạc |
| tonnetz_std | unitless | <0.05 | 0.05 – 0.2 | >0.2 | --- |
| rms_mean | 0–1 (norm) | <0.01 | 0.01 – 0.1 | >0.1 | Biểu diễn năng lượng |
| rms_std | 0–1 | <0.005 | 0.005 – 0.05 | >0.05 | --- |
| rms_max | 0–1 | <0.05 | 0.05 – 0.3 | >0.3 | Đỉnh năng lượng |
| rms_min | 0–1 | <0.001 | 0.001 – 0.01 | >0.01 | --- |
| tempo | BPM | <60 | 60 – 140 | >140 | Chủ yếu cho nhạc |
| duration | s | <1 | 1 – 30 | >30 | Độ dài file |
| loudness | dB (relative) | < -50 | -50 – -20 | > -20 | Gần 0 dB → clipping |
| loudness_peak | dB | < -50 | -50 – -20 | > -20 | --- |
| spectral_bandwidth_mean | Hz | <500 | 500 – 2000 | >2000 | Phân bố năng lượng |
| spectral_bandwidth_std | Hz | <100 | 100 – 500 | >500 | --- |
| spectral_flatness_mean | 0–1 | <0.1 | 0.1 – 0.5 | >0.5 | 0 = tonal, 1 = noise |
| spectral_flatness_std | 0–1 | <0.02 | 0.02 – 0.1 | >0.1 | --- |
| hnr | dB | <5 | 5 – 20 | >20 | HNR thấp → nhiều nhiễu |
| spectral_slope_mean | unit | (negative typical) | (moderate) | (steep) | Tùy cách tính (dB/Hz hoặc dB/oct) |
| spectral_slope_std | same | <0.5 | 0.5 – 2 | >2 | --- |
| spectral_kurtosis_mean | unitless | small | medium | large | Giá trị phụ thuộc chuẩn hóa |
| spectral_kurtosis_std | unitless | <1 | 1 – 5 | >5 | --- |
| spectral_skewness_mean | unitless | negative/near0/positive | - | - | Negative → nhiều cao tần |
| spectral_skewness_std | unitless | <0.2 | 0.2 – 1 | >1 | --- |
| onset_strength_mean | unit | <0.01 | 0.01 – 0.1 | >0.1 | Nhạc percussive cao hơn thoại |
| onset_strength_std | unit | <0.01 | 0.01 – 0.05 | >0.05 | --- |
| spectral_flux | unit | <0.01 | 0.01 – 0.1 | >0.1 | Thay đổi phổ theo thời gian |

### Ghi chú
- Các ngưỡng trên mang tính tham khảo; khuyến nghị tính thống kê (min/median/75p/iqr) trên toàn dataset để tinh chỉnh ngưỡng phù hợp.
- MFCC/tonnetz/kurtosis/skewness/slope thường cần chuẩn hóa (z-score) trước khi dùng làm đặc trưng cho mô hình.
- Nếu muốn, có thể thêm script tính ngưỡng tự động (ví dụ: low = quantile(0.10), medium = 10–90% range, high = quantile(0.90)).

### Giải thích chi tiết
1. Pitch (Độ cao)
Tên đầy đủ: Pitch

Dịch nghĩa: Độ cao hay Cao độ giọng.

Giải thích: Đây là đặc trưng cơ bản nhất của âm thanh, thể hiện tần số cơ bản (F0) của giọng nói hoặc nốt nhạc, cho biết âm thanh đó trầm hay cao. Nó được cảm nhận bởi tai người. Ví dụ, giọng nam thường có pitch thấp hơn giọng nữ.

Mean (Trung bình): Độ cao trung bình của toàn bộ đoạn âm thanh.

Std (Độ lệch chuẩn): Mức độ biến thiên về độ cao. Std cao cho thấy giọng nói/giai điệu có nhiều ngữ điệu lên xuống.

Range (Biên độ): Chênh lệch giữa pitch cao nhất và thấp nhất.

2. Spectral Centroid (Tâm phổ)
Tên đầy đủ: Spectral Centroid

Dịch nghĩa: Tâm phổ hoặc Độ sáng/tối của âm thanh.

Giải thích: Đặc trưng này xác định "trọng tâm" của phổ tín hiệu âm thanh. Hãy tưởng tượng phổ tần số là một hình dạng, Spectral Centroid chính là điểm cân bằng của hình dạng đó.

Giá trị cao tương ứng với âm thanh "sáng" hơn, "sắc" hơn (ví dụ: tiếng cymbal, giọng nói a, i).

Giá trị thấp tương ứng với âm thanh "tối" hơn, "trầm" hơn, "ấm" hơn (ví dụ: tiếng trống bass, giọng nói u, o).

Mean, Std: Phản ánh độ sáng trung bình và sự thay đổi về độ sáng của âm thanh.

3. Spectral Rolloff (Ngưỡng lăn phổ)
Tên đầy đủ: Spectral Rolloff

Dịch nghĩa: Ngưỡng lăn phổ hoặc Độ rõ ràng của âm thanh.

Giải thích: Đây là tần số mà dưới nó chứa một tỷ lệ phần trăm nhất định (thường là 85% hoặc 95%) của tổng năng lượng phổ. Nó giúp phân biệt âm thanh có cấu trúc hài hòa (nhiều năng lượng ở tần số thấp, rolloff thấp) và âm thanh nhiễu (năng lượng trải đều, rolloff cao).

Mean, Std: Cho biết ngưỡng năng lượng phổ trung bình và sự biến thiên của nó.

4. Zero Crossing Rate (Tốc độ qua điểm 0)
Tên đầy đủ: Zero-Crossing Rate (ZCR)

Dịch nghĩa: Tốc độ qua điểm không.

Giải thích: Là số lần tín hiệu âm thanh (sóng âm) đi qua trục hoành (giá trị 0) trong một khoảng thời gian.

ZCR cao thường xuất hiện ở các âm thanh có nhiều tần số cao hoặc nhiễu, ví dụ như phụ âm xát ("s", "sh").

ZCR thấp thường xuất hiện ở các âm thanh có tính chu kỳ, du dương như nguyên âm ("a", "o") hoặc nhạc cụ.

Đặc trưng này rất hữu ích trong việc phân biệt giữa giọng nói (voiced sound) và âm thanh không lời (unvoiced sound).

5. Chroma Features (Đặc trưng Sắc độ)
Tên đầy đủ: Chromagram hoặc Chroma Features

Dịch nghĩa: Đặc trưng Sắc độ hoặc Véc-tơ Sắc độ.

Giải thích: Đặc trưng này chiếu toàn bộ phổ tần số vào 12 thùng (bins) tương ứng với 12 nốt nhạc trong thang âm Tây phương (C, C#, D, D#, E, F, F#, G, G#, A, A#, B). Nó rất hữu ích trong phân tích âm nhạc vì nó không phụ thuộc vào quãng tám (octave), chỉ tập trung vào "sắc thái" của nốt nhạc. Ví dụ, nốt Đô ở các quãng tám khác nhau đều được gom vào cùng một thùng "C".

Mean, Std: Cho biết sự phân bổ trung bình của các nốt nhạc và sự thay đổi của chúng trong đoạn nhạc.

6. Spectral Contrast (Độ tương phản phổ)
Tên đầy đủ: Spectral Contrast

Dịch nghĩa: Độ tương phản phổ.

Giải thích: Đo lường sự khác biệt về biên độ giữa các đỉnh (peaks) và các đáy (valleys) trong phổ tần số.

Độ tương phản cao cho thấy sự khác biệt rõ rệt giữa các thành phần tần số, thường gặp trong âm nhạc có cấu trúc rõ ràng.

Độ tương phản thấp cho thấy phổ phẳng hơn, thường gặp trong các tín hiệu nhiễu.

Mean, Std: Phản ánh độ tương phản trung bình và sự thay đổi của nó.

7. Tonnetz (Mạng lưới τονικότητα)
Tên đầy đủ: Tonal Centroid Features (Tonnetz)

Dịch nghĩa: Đặc trưng trọng tâm τονικότητα.

Giải thích: Đây là một đặc trưng cao cấp hơn Chroma, thể hiện mối quan hệ hài hòa giữa các nốt nhạc dựa trên lý thuyết âm nhạc (vòng tròn bậc năm - circle of fifths). Nó hữu ích để phân tích cấu trúc hợp âm và sự chuyển điệu trong âm nhạc.

8. RMS Energy (Năng lượng RMS)
Tên đầy đủ: Root Mean Square Energy

Dịch nghĩa: Năng lượng trung bình bình phương.

Giải thích: Đo lường biên độ (amplitude) của tín hiệu, liên quan trực tiếp đến độ to mà tai người cảm nhận được.

Mean (Trung bình): Năng lượng trung bình của tín hiệu.

Std (Độ lệch chuẩn): Mức độ thay đổi về năng lượng (độ to).

Max, Min: Năng lượng tại điểm to nhất và nhỏ nhất.

9. Tempo (Nhịp độ)
Tên đầy đủ: Tempo

Dịch nghĩa: Nhịp độ.

Giải thích: Tốc độ của bản nhạc, thường được đo bằng số phách mỗi phút (Beats Per Minute - BPM).

10. Duration (Thời lượng)
Tên đầy đủ: Duration

Dịch nghĩa: Thời lượng.

Giải thích: Độ dài của đoạn âm thanh, tính bằng giây.

11. Loudness (Độ to)
Tên đầy đủ: Loudness

Dịch nghĩa: Độ to.

Giải thích: Mức độ âm thanh được cảm nhận, thường được đo bằng decibel (dB). Nó liên quan đến RMS Energy nhưng được biểu diễn trên thang đo logarit, gần với cách tai người nghe.

12. Spectral Bandwidth (Độ rộng băng thông phổ)
Tên đầy đủ: Spectral Bandwidth

Dịch nghĩa: Độ rộng băng thông phổ.

Giải thích: Đo lường "bề rộng" của phổ tần số xung quanh tâm phổ (Spectral Centroid).

Băng thông rộng cho thấy tín hiệu có nhiều thành phần tần số khác nhau (ví dụ: tiếng nhiễu trắng).

Băng thông hẹp cho thấy năng lượng tập trung ở một vài tần số nhất định (ví dụ: tiếng sáo đơn).

13. Spectral Flatness (Độ phẳng phổ)
Tên đầy đủ: Spectral Flatness

Dịch nghĩa: Độ phẳng phổ.

Giải thích: Đo lường mức độ "phẳng" hoặc "gồ ghề" của phổ tần số.

Giá trị gần 1.0 cho thấy phổ rất phẳng, giống như nhiễu trắng (white noise), năng lượng phân bổ đều.

Giá trị gần 0.0 cho thấy phổ có nhiều đỉnh nhọn, có nghĩa là âm thanh có tính giai điệu rõ ràng.

14. Harmonic-to-Noise Ratio (Tỷ lệ hài hòa/nhiễu)
Tên đầy đủ: Harmonic-to-Noise Ratio (HNR)

Dịch nghĩa: Tỷ lệ hài hòa trên nhiễu.

Giải thích: Đo lường tỷ lệ giữa năng lượng của các thành phần hài hòa (có tính chu kỳ, du dương) và năng lượng của các thành phần nhiễu (không có tính chu kỳ).

HNR cao cho thấy âm thanh trong, rõ ràng, có tính nhạc cao (ví dụ: giọng hát tốt, tiếng violin).

HNR thấp cho thấy âm thanh có nhiều tạp âm, tiếng thở, hoặc bị rè.

15. Spectral Slope (Độ dốc phổ)
Tên đầy đủ: Spectral Slope

Dịch nghĩa: Độ dốc phổ.

Giải thích: Mô tả độ dốc của đường hồi quy tuyến tính trên phổ tần số, cho thấy năng lượng phổ giảm nhanh hay chậm khi tần số tăng. Nó liên quan đến đặc tính của nguồn phát âm.

16. Spectral Kurtosis (Độ nhọn phổ)
Tên đầy đủ: Spectral Kurtosis

Dịch nghĩa: Độ nhọn phổ.

Giải thích: Đo lường mức độ "nhọn" hoặc "bằng" của phân bố phổ so với phân bố chuẩn (Gaussian). Nó cho biết sự hiện diện của các đỉnh bất thường trong phổ.

17. Spectral Skewness (Độ lệch phổ)
Tên đầy đủ: Spectral Skewness

Dịch nghĩa: Độ lệch phổ.

Giải thích: Đo lường mức độ bất đối xứng của phân bố phổ. Nó cho biết liệu phần lớn năng lượng tập trung ở bên trái (tần số thấp) hay bên phải (tần số cao) của giá trị trung bình.

18. Onset Strength (Độ mạnh khởi âm)
Tên đầy đủ: Onset Strength

Dịch nghĩa: Độ mạnh khởi âm.

Giải thích: Đo lường sự thay đổi năng lượng phổ theo thời gian để phát hiện các điểm khởi đầu của một nốt nhạc hoặc một âm thanh (gọi là "onset"). Giá trị này sẽ tăng vọt tại thời điểm một nốt nhạc mới được gảy hoặc một âm tiết mới được phát ra.

19. Spectral Flux (Dòng phổ)
Tên đầy đủ: Spectral Flux

Dịch nghĩa: Dòng phổ hoặc Luồng phổ.

Giải thích: Đo lường tốc độ thay đổi của phổ tần số giữa các khung thời gian liên tiếp. Giá trị cao cho thấy âm sắc (timbre) của âm thanh đang thay đổi nhanh.

🎵 MFCC (Mel-Frequency Cepstral Coefficients)
Tên đầy đủ: Mel-Frequency Cepstral Coefficients

Dịch nghĩa: Các hệ số Cepstrum trên thang Mel.

Giải thích tổng quan: Đây là một trong những đặc trưng quan trọng và mạnh mẽ nhất trong xử lý tiếng nói và âm thanh. Nó mô tả hình dạng tổng thể của phổ tín hiệu (spectral envelope) theo một cách rất giống với cách tai người cảm nhận âm thanh (sử dụng thang đo Mel). Về cơ bản, MFCC là "dấu vân tay" (fingerprint) của âm sắc.

Ý nghĩa của 13 hệ số MFCC
Quan trọng nhất cần hiểu là: Không có một khoảng giá trị cố định nào để đánh giá một hệ số MFCC là "tốt" hay "xấu". Giá trị của chúng có ý nghĩa khi được so sánh với nhau và được dùng làm đầu vào cho các mô hình học máy. Mô hình sẽ học cách nhận diện các mẫu (pattern) từ chuỗi 13 hệ số này để phân loại âm thanh.

Dưới đây là ý nghĩa tương đối của từng hệ số:

Hệ số 0 (MFCC 0):

Đại diện cho: Năng lượng tổng thể (overall energy) hoặc độ to của tín hiệu trong một khung thời gian ngắn.

Giải thích: Nó tương tự như RMS Energy nhưng trên thang logarit. Hệ số này thường được loại bỏ trong một số ứng dụng (như nhận dạng giọng nói) vì độ to của âm thanh có thể thay đổi tùy vào khoảng cách đến micro, không phản ánh nội dung lời nói.

Hệ số 1 đến 12 (MFCC 1-12):

Đại diện cho: Hình dạng của phổ tần số. Chúng chứa thông tin về âm sắc (timbre), giúp phân biệt các âm thanh khác nhau (ví dụ: nguyên âm "a" và "i", hoặc tiếng đàn guitar và piano).

Phân chia vai trò:

Các hệ số bậc thấp (MFCC 1, 2, 3...): Mô tả hình dạng tổng quan, thô của phổ. Chúng rất quan trọng trong việc nhận dạng các nguyên âm, vì mỗi nguyên âm có một cấu trúc formant (các đỉnh năng lượng trong phổ) đặc trưng.

Các hệ số bậc cao (MFCC 7, 8, 9...): Mô tả các chi tiết tinh vi, nhỏ hơn của phổ. Chúng có thể chứa thông tin về các hài âm cao hơn, kết cấu (texture) của âm thanh, hoặc thậm chí là đặc điểm riêng của người nói/nhạc cụ.



## Lưu ý
Chạy một batch đầu (150 file): python train_audio_features.py
Chạy step 2: python train_audio_features.py --step 2
Chạy tất cả tuần tự và ghi master: python train_audio_features.py --all --append-master
Chạy tất cả bắt đầu từ step 4: python train_audio_features.py --all --from-step 4