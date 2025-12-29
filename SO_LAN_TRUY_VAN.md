# SỐ LẦN TRUY VẤN KHI TRẢ VỀ KẾT QUẢ

## PHÂN TÍCH CHI TIẾT

### 1. Quy trình tìm kiếm (Hàm `search_similar_voices`)

**File**: `voice_search_gui_enhanced.py`, dòng 488-602

---

## 2. CÁC LOẠI TRUY VẤN

### 2.1. Trích xuất đặc trưng (Feature Extraction)

**Số lần**: **1 lần**

**Vị trí**: Dòng 517
```python
features = self.extract_audio_features(audio_path, progress_callback)
```

**Chi tiết**:
- Load file audio: 1 lần
- Trích xuất tất cả features: 1 lần (50+ features)
- Tính toán: RMS, MFCC, Spectral, Pitch, etc.

**Thời gian**: Phụ thuộc vào độ dài file audio

---

### 2.2. KNN Query (Tìm kiếm K-Nearest Neighbors)

**Số lần**: **1 lần**

**Vị trí**: Dòng 548
```python
distances, indices = self.knn_model.kneighbors(
    feature_vector_scaled, 
    n_neighbors=min(search_k, len(self.df_train))
)
```

**Chi tiết**:
- `search_k = k * 3` nếu có filter dialect, ngược lại `search_k = k`
- Mặc định: `k = 10` → `search_k = 10` hoặc `30`
- Trả về: `search_k` neighbors gần nhất

**Ví dụ**:
- Không filter: `k=10` → tìm 10 neighbors
- Có filter: `k=10` → tìm 30 neighbors (để đảm bảo đủ sau khi lọc)

**Thời gian**: O(N) với brute force, N = số mẫu training (3906 mẫu)

---

### 2.3. Truy vấn Training Data

**Số lần**: **search_k lần** (trong vòng lặp)

**Vị trí**: Dòng 552-553
```python
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    sample = self.df_train.iloc[idx]  # ← Truy vấn training data
```

**Chi tiết**:
- Mỗi lần lặp: Truy vấn 1 mẫu từ DataFrame
- Tổng cộng: `search_k` lần truy vấn
- **Lưu ý**: DataFrame đã load sẵn trong memory → O(1) mỗi lần

**Ví dụ**:
- `search_k = 10` → 10 lần truy vấn
- `search_k = 30` → 30 lần truy vấn

---

### 2.4. Truy vấn Speaker Database

**Số lần**: **Tối đa 2 × search_k lần** (cho mỗi kết quả)

**Vị trí**: Dòng 571-578
```python
if self.speaker_db is not None:
    speaker_info = self.speaker_db[self.speaker_db['speaker_id'] == speaker_id]  # ← Query 1
    if not speaker_info.empty:
        speaker_name = speaker_info.iloc[0]['vietnamese_name']
    else:
        speaker_info = self.speaker_db[self.speaker_db['dialect'] == speaker_id]  # ← Query 2
        if not speaker_info.empty:
            speaker_name = speaker_info.iloc[0]['vietnamese_name']
```

**Chi tiết**:
- Query 1: Tìm theo `speaker_id` → 1 lần cho mỗi kết quả
- Query 2: Nếu không tìm thấy, tìm theo `dialect` → thêm 1 lần
- Tổng cộng: 1-2 lần cho mỗi kết quả

**Ví dụ**:
- `search_k = 10` → Tối đa 20 lần truy vấn speaker database
- `search_k = 30` → Tối đa 60 lần truy vấn speaker database

---

### 2.5. Tính toán Similarity

**Số lần**: **search_k lần** (cho mỗi neighbor)

**Vị trí**: Dòng 554, 560-563
```python
base_similarity = (1 - dist) * 100  # ← Tính similarity
similarity = base_similarity
if boost_same_dialect and ...:
    similarity = min(100, base_similarity + 20)  # ← Boost nếu cần
```

**Chi tiết**:
- Tính base similarity: 1 lần cho mỗi neighbor
- Boost similarity: 0-1 lần (nếu điều kiện thỏa mãn)
- Tổng cộng: `search_k` lần tính toán

---

### 2.6. Lấy Features của mỗi kết quả

**Số lần**: **search_k lần** (cho mỗi kết quả)

**Vị trí**: Dòng 581-583
```python
sample_features = {}
for col in feature_cols:
    sample_features[col] = sample.get(col, 0.0)  # ← Lấy từng feature
```

**Chi tiết**:
- Lặp qua tất cả feature columns (63 features)
- Tổng cộng: `search_k × 63` lần truy cập dữ liệu

---

## 3. TỔNG KẾT SỐ LẦN TRUY VẤN

### 3.1. Trường hợp không filter (k=10)

| Loại truy vấn | Số lần | Chi tiết |
|---------------|--------|----------|
| **Extract features** | **1** | Trích xuất từ input audio |
| **KNN query** | **1** | Tìm 10 neighbors |
| **Training data** | **10** | Lấy 10 mẫu từ DataFrame |
| **Speaker database** | **10-20** | Tìm tên người nói (1-2 lần/mẫu) |
| **Tính similarity** | **10** | Tính cho 10 neighbors |
| **Lấy features** | **630** | 10 mẫu × 63 features |
| **TỔNG CỘNG** | **662-672** | |

### 3.2. Trường hợp có filter (k=10, filter_dialect=True)

| Loại truy vấn | Số lần | Chi tiết |
|---------------|--------|----------|
| **Extract features** | **1** | Trích xuất từ input audio |
| **KNN query** | **1** | Tìm 30 neighbors (k×3) |
| **Training data** | **30** | Lấy 30 mẫu từ DataFrame |
| **Speaker database** | **30-60** | Tìm tên người nói (1-2 lần/mẫu) |
| **Tính similarity** | **30** | Tính cho 30 neighbors |
| **Lấy features** | **1890** | 30 mẫu × 63 features |
| **Filter dialect** | **30** | Kiểm tra dialect cho 30 mẫu |
| **TỔNG CỘNG** | **1982-2012** | (Chỉ lấy 10 kết quả cuối) |

---

## 4. PHÂN TÍCH HIỆU SUẤT

### 4.1. Truy vấn tốn thời gian nhất

1. **KNN Query** (1 lần): O(N) với brute force
   - N = 3906 mẫu training
   - Tính cosine distance với tất cả mẫu
   - **Thời gian**: ~0.1-1 giây (tùy CPU)

2. **Extract Features** (1 lần): Phụ thuộc độ dài audio
   - Xử lý audio file
   - Tính 50+ features
   - **Thời gian**: ~1-5 giây (tùy độ dài file)

3. **Lấy Features** (630-1890 lần): Truy cập memory
   - DataFrame đã load sẵn
   - **Thời gian**: ~0.01-0.1 giây (rất nhanh)

### 4.2. Truy vấn nhanh

- **Training data**: O(1) mỗi lần (đã load sẵn)
- **Speaker database**: O(1) mỗi lần (đã load sẵn)
- **Tính similarity**: O(1) mỗi lần (phép tính đơn giản)

---

## 5. TỐI ƯU HÓA

### 5.1. Giảm số lần truy vấn

**Hiện tại**:
- KNN query: 1 lần (tốt)
- Training data: search_k lần (có thể tối ưu)
- Speaker database: 2×search_k lần (có thể tối ưu)

**Cải tiến**:
- **Batch query**: Lấy tất cả samples cùng lúc thay vì từng cái
- **Index speaker database**: Tạo dictionary để O(1) lookup
- **Cache features**: Lưu features đã tính để tránh tính lại

### 5.2. Tối ưu KNN

**Hiện tại**: Brute force O(N)
- Tìm kiếm trong 3906 mẫu
- Tính cosine distance với tất cả

**Cải tiến**:
- **Ball Tree/KD-Tree**: O(log N)
- **LSH**: O(1) trung bình
- **FAISS**: Cực kỳ nhanh với GPU

---

## 6. KẾT LUẬN

### Số lần truy vấn chính:

| Trường hợp | KNN Query | Training Data | Speaker DB | Tổng |
|------------|-----------|---------------|------------|------|
| **Không filter (k=10)** | 1 | 10 | 10-20 | **~662-672** |
| **Có filter (k=10)** | 1 | 30 | 30-60 | **~1982-2012** |

### Truy vấn quan trọng nhất:

1. **KNN Query**: **1 lần** - Tốn thời gian nhất (O(N))
2. **Extract Features**: **1 lần** - Tốn thời gian thứ hai
3. **Các truy vấn khác**: Nhiều lần nhưng rất nhanh (O(1))

### Tổng thời gian:

- **Không filter**: ~1-6 giây
- **Có filter**: ~1-6 giây (tương tự, chỉ tìm nhiều neighbors hơn)

**Lưu ý**: Hầu hết thời gian dành cho KNN query và extract features, các truy vấn khác rất nhanh vì đã load sẵn trong memory.

