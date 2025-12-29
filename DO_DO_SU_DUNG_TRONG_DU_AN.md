# CÁC ĐỘ ĐO ĐƯỢC SỬ DỤNG TRONG DỰ ÁN

## 1. ĐỘ ĐO CHÍNH: COSINE SIMILARITY

### 1.1. Vị trí trong code

**File**: `voice_search_gui_enhanced.py`

**Dòng 144**: Khởi tạo KNN model với Cosine Similarity
```python
self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
```

**Dòng 158**: Lưu metric vào training info
```python
'metric': 'cosine',
```

**File**: `trained_model.json`
```json
"metric": "cosine"
```

### 1.2. Cách hoạt động

**Cosine Similarity** đo độ tương tự dựa trên góc giữa hai vector trong không gian đặc trưng.

**Công thức**:
```
cosine_similarity = (A · B) / (||A|| × ||B||)
```

**Trong code** (dòng 548, 554):
```python
# Tìm K nearest neighbors (trả về distances)
distances, indices = self.knn_model.kneighbors(feature_vector_scaled, ...)

# Chuyển đổi distance → similarity score (0-100%)
base_similarity = (1 - dist) * 100
```

**Giải thích**:
- KNN với `metric='cosine'` trả về **cosine distance** (1 - cosine_similarity)
- Cosine distance có giá trị từ 0 (giống nhau) đến 2 (khác nhau hoàn toàn)
- Chuyển đổi: `similarity = (1 - distance) × 100` để có score từ 0-100%

### 1.3. Tại sao chọn Cosine Similarity?

**Ưu điểm**:
- ✅ Không phụ thuộc vào độ lớn của vector, chỉ phụ thuộc vào hướng
- ✅ Phù hợp với dữ liệu đã được chuẩn hóa (StandardScaler)
- ✅ Hiệu quả với dữ liệu thưa (sparse data)
- ✅ Phù hợp với audio features (MFCC, spectral features)

**Nhược điểm**:
- ⚠️ Không tính đến độ lớn của vector
- ⚠️ Có thể không phù hợp với một số loại đặc trưng

---

## 2. CÁC ĐỘ ĐO KHÁC ĐƯỢC SỬ DỤNG

### 2.1. Similarity Score (Từ Distance)

**Vị trí**: Dòng 554
```python
base_similarity = (1 - dist) * 100
```

**Mục đích**: Chuyển đổi distance thành similarity score dễ hiểu (0-100%)

**Công thức**:
```
similarity_score = (1 - cosine_distance) × 100
```

**Kết quả**:
- `distance = 0` → `similarity = 100%` (giống nhau hoàn toàn)
- `distance = 1` → `similarity = 0%` (khác nhau hoàn toàn)
- `distance = 0.2` → `similarity = 80%` (tương tự)

### 2.2. Boost Similarity (Cho cùng vùng miền)

**Vị trí**: Dòng 560-563
```python
similarity = base_similarity
if boost_same_dialect and filter_dialect and sample_dialect == filter_dialect:
    similarity = min(100, base_similarity + 20)  # Boost +20%
elif boost_same_dialect and self.input_dialect and sample_dialect == self.input_dialect:
    similarity = min(100, base_similarity + 20)  # Boost +20%
```

**Mục đích**: Tăng similarity score cho các mẫu cùng vùng miền

**Công thức**:
```
boosted_similarity = min(100, base_similarity + 20)
```

**Ví dụ**:
- `base_similarity = 75%` → `boosted_similarity = 95%` (nếu cùng dialect)
- `base_similarity = 90%` → `boosted_similarity = 100%` (capped at 100%)

### 2.3. Feature Similarity (So sánh từng feature)

**Vị trí**: Dòng 1277-1279
```python
similarity_pct = max(0, (1 - diff / avg_val) * 100)
```

**Mục đích**: So sánh từng feature riêng lẻ giữa input và kết quả

**Công thức**:
```
feature_similarity = max(0, (1 - |input_feature - result_feature| / avg_value) × 100)
```

**Giải thích**:
- Tính độ khác biệt giữa 2 giá trị feature
- Chia cho giá trị trung bình để normalize
- Chuyển thành percentage similarity

**Ngưỡng** (dòng 1282-1292):
- `> 85%`: Rất giống (màu xanh lá)
- `> 70%`: Giống (màu xanh dương)
- `> 50%`: Tương đối giống (màu vàng)
- `≤ 50%`: Khác (màu đỏ)

---

## 3. TÓM TẮT CÁC ĐỘ ĐO

| Độ đo | Vị trí | Công thức | Mục đích | **Tầm quan trọng** |
|-------|--------|-----------|----------|-------------------|
| **Cosine Similarity** | Dòng 144 | `cos(A,B) = (A·B)/(\|A\|\|B\|)` | Độ đo chính cho KNN | **100%** (Chủ yếu) |
| **Cosine Distance** | Dòng 548 | `distance = 1 - cosine_similarity` | Khoảng cách giữa 2 vector | **100%** (Tính từ Cosine) |
| **Similarity Score** | Dòng 554 | `(1 - dist) × 100` | Chuyển distance → % | **100%** (Hiển thị kết quả) |
| **Boost Similarity** | Dòng 560-563 | `min(100, base + 20)` | Tăng score cho cùng dialect | **20%** (Điều chỉnh) |
| **Feature Similarity** | Dòng 1277 | `(1 - diff/avg) × 100` | So sánh từng feature | **0%** (Chỉ debug) |

### 3.1. Phần trăm tầm quan trọng

**Cosine Similarity**: **100%**
- Là độ đo gốc, tất cả tính toán dựa trên nó
- Được sử dụng trong toàn bộ quá trình tìm kiếm
- Không thể thay thế trong KNN model

**Cosine Distance**: **100%** (phụ thuộc Cosine Similarity)
- Tính trực tiếp từ Cosine Similarity: `distance = 1 - similarity`
- Là output của KNN model
- 100% phụ thuộc vào Cosine Similarity

**Similarity Score**: **100%** (hiển thị)
- Chuyển đổi từ Cosine Distance
- Được hiển thị trong tất cả kết quả tìm kiếm
- 100% kết quả đều dùng Similarity Score

**Boost Similarity**: **~20%** (điều chỉnh có điều kiện)
- Chỉ áp dụng khi: cùng vùng miền VÀ bật boost
- Tăng thêm tối đa 20% so với base similarity
- Không phải độ đo gốc, chỉ là điều chỉnh

**Feature Similarity**: **0%** (không dùng trong tìm kiếm)
- Chỉ dùng để so sánh chi tiết trong debug mode
- Không ảnh hưởng đến kết quả tìm kiếm
- Chỉ để hiển thị thông tin

### 3.2. Phần trăm giá trị trong kết quả

**Cosine Similarity**: 0.0 - 1.0 (0% - 100%)
- Giá trị thực tế: 0.0 (khác nhau) đến 1.0 (giống nhau)
- Được chuyển thành Similarity Score để hiển thị

**Similarity Score**: 0% - 100%
- **0-60%**: Thấp (màu đỏ) - Không tương tự
- **60-80%**: Trung bình (màu cam) - Tương đối tương tự
- **80-100%**: Cao (màu xanh lá) - Rất tương tự

**Boost Similarity**: +0% đến +20%
- Chỉ áp dụng khi điều kiện thỏa mãn
- Tăng thêm tối đa 20% so với base similarity
- Ví dụ: 75% → 95% (nếu boost), 90% → 100% (capped)

**Feature Similarity**: 0% - 100%
- So sánh từng feature riêng lẻ
- Ngưỡng: >85% (rất giống), >70% (giống), >50% (tương đối)

---

## 4. ĐỘ ĐO CHỦ YẾU

### ✅ **COSINE SIMILARITY** là độ đo chủ yếu

**Lý do**:
1. **Được sử dụng trong KNN model**: Metric chính để tìm kiếm
2. **Được lưu trong model**: `trained_model.json` ghi nhận `"metric": "cosine"`
3. **Tất cả tính toán dựa trên nó**: Distance từ Cosine → Similarity Score
4. **Phù hợp với dữ liệu**: Audio features đã được chuẩn hóa

**Các độ đo khác**:
- **Similarity Score**: Chỉ là cách biểu diễn Cosine Similarity dưới dạng %
- **Boost Similarity**: Chỉ là điều chỉnh thêm, không phải độ đo gốc
- **Feature Similarity**: Chỉ dùng để so sánh chi tiết, không dùng trong tìm kiếm

---

## 5. CÁC ĐỘ ĐO CÓ THỂ THAY THẾ (Chưa sử dụng)

### 5.1. Euclidean Distance
- **Metric**: `'euclidean'`
- **Công thức**: `√(Σ(Ai - Bi)²)`
- **Ưu điểm**: Dễ hiểu, phù hợp dữ liệu liên tục
- **Nhược điểm**: Bị ảnh hưởng bởi scale

### 5.2. Manhattan Distance
- **Metric**: `'manhattan'`
- **Công thức**: `Σ|Ai - Bi|`
- **Ưu điểm**: Ít nhạy cảm với outliers
- **Nhược điểm**: Kém hiệu quả với high-dim

### 5.3. Correlation Distance
- **Metric**: `'correlation'`
- **Công thức**: `1 - correlation_coefficient`
- **Ưu điểm**: Đo tương quan tuyến tính
- **Nhược điểm**: Chỉ phát hiện quan hệ tuyến tính

**Lưu ý**: Các metric này có thể thay đổi trong code bằng cách sửa dòng 144:
```python
# Thay đổi từ:
self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')

# Thành:
self.knn_model = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='brute')
# hoặc 'manhattan', 'correlation'
```

---

## 6. KẾT LUẬN

### Độ đo chủ yếu: **COSINE SIMILARITY**

**Trong dự án**:
- ✅ Sử dụng **Cosine Similarity** làm metric chính cho KNN
- ✅ Chuyển đổi thành **Similarity Score** (0-100%) để dễ hiểu
- ✅ Có **Boost Similarity** (+20%) cho cùng vùng miền
- ✅ Có **Feature Similarity** để so sánh chi tiết từng feature

**Công thức tổng quát**:
```
1. Cosine Distance = 1 - Cosine Similarity
2. Base Similarity = (1 - Cosine Distance) × 100
3. Final Similarity = min(100, Base Similarity + Boost)
```

**Vị trí code**:
- **Khởi tạo**: Dòng 144 (`metric='cosine'`)
- **Tính toán**: Dòng 548 (`kneighbors()`)
- **Chuyển đổi**: Dòng 554 (`(1 - dist) * 100`)
- **Boost**: Dòng 560-563 (`+20%`)

