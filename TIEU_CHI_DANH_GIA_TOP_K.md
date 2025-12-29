# CÁC TIÊU CHÍ ĐÁNH GIÁ TOP K GIỐNG NHAU NHẤT

## 1. TỔNG QUAN

Hệ thống sử dụng nhiều tiêu chí để xác định và sắp xếp top K giọng nói tương tự nhất. Các tiêu chí được áp dụng theo thứ tự ưu tiên.

**File**: `voice_search_gui_enhanced.py`, hàm `search_similar_voices()` (dòng 488-602)

---

## 2. CÁC TIÊU CHÍ CHÍNH

### 2.1. Tiêu chí 1: Cosine Distance (Ưu tiên cao nhất)

**Vị trí**: Dòng 548
```python
distances, indices = self.knn_model.kneighbors(feature_vector_scaled, n_neighbors=min(search_k, len(self.df_train)))
```

**Mô tả**:
- **Cosine Distance** là tiêu chí chính và quan trọng nhất
- Được tính toán bởi KNN model với `metric='cosine'`
- KNN trả về các neighbors đã được **sắp xếp theo distance tăng dần** (từ nhỏ đến lớn)
- Distance nhỏ hơn = giống nhau hơn

**Công thức**:
```
cosine_distance = 1 - cosine_similarity
cosine_similarity = (A · B) / (||A|| × ||B||)
```

**Giá trị**:
- `distance = 0` → Giống nhau hoàn toàn
- `distance = 1` → Khác nhau hoàn toàn
- `distance = 0.2` → Tương đối giống (80% similarity)

**Ưu tiên**: **CAO NHẤT** - Tất cả các tiêu chí khác đều dựa trên tiêu chí này

---

### 2.2. Tiêu chí 2: Base Similarity Score

**Vị trí**: Dòng 554
```python
base_similarity = (1 - dist) * 100
```

**Mô tả**:
- Chuyển đổi Cosine Distance thành Similarity Score (0-100%)
- Dễ hiểu và hiển thị hơn so với distance
- Giữ nguyên thứ tự sắp xếp từ KNN

**Công thức**:
```
base_similarity = (1 - cosine_distance) × 100
```

**Giá trị**:
- `distance = 0` → `similarity = 100%` (giống nhau hoàn toàn)
- `distance = 0.2` → `similarity = 80%` (tương đối giống)
- `distance = 1` → `similarity = 0%` (khác nhau hoàn toàn)

**Ưu tiên**: **CAO** - Là cơ sở cho tất cả tính toán sau

---

### 2.3. Tiêu chí 3: Boost Similarity (Điều chỉnh)

**Vị trí**: Dòng 560-563
```python
similarity = base_similarity
if boost_same_dialect and filter_dialect and sample_dialect == filter_dialect:
    similarity = min(100, base_similarity + 20)  # Boost +20%
elif boost_same_dialect and self.input_dialect and sample_dialect == self.input_dialect:
    similarity = min(100, base_similarity + 20)  # Boost +20%
```

**Mô tả**:
- Tăng similarity score thêm **20%** cho các mẫu cùng vùng miền
- Chỉ áp dụng khi:
  - Bật boost (`boost_same_dialect = True`)
  - VÀ (cùng dialect với filter HOẶC cùng dialect với input)

**Công thức**:
```
if (cùng_dialect):
    final_similarity = min(100, base_similarity + 20)
else:
    final_similarity = base_similarity
```

**Ví dụ**:
- `base_similarity = 75%` → `final_similarity = 95%` (nếu boost)
- `base_similarity = 90%` → `final_similarity = 100%` (capped at 100%)

**Ưu tiên**: **TRUNG BÌNH** - Chỉ điều chỉnh, không thay đổi thứ tự gốc từ KNN

**Lưu ý**: Boost có thể làm thay đổi thứ tự ranking nếu có nhiều mẫu cùng base_similarity

---

### 2.4. Tiêu chí 4: Filter theo Vùng miền

**Vị trí**: Dòng 565-567
```python
if filter_dialect and sample_dialect != filter_dialect:
    continue  # Bỏ qua mẫu này
```

**Mô tả**:
- Lọc bỏ các mẫu không cùng vùng miền nếu có filter
- Chỉ giữ lại các mẫu có `dialect == filter_dialect`
- Áp dụng sau khi tính similarity

**Điều kiện**:
- `filter_dialect != None` (người dùng chọn vùng miền cụ thể)
- `sample_dialect != filter_dialect` → Bỏ qua

**Ưu tiên**: **CAO** - Quyết định mẫu nào được giữ lại

**Lưu ý**: 
- Tìm `search_k = k × 3` neighbors để đảm bảo đủ kết quả sau khi lọc
- Nếu không đủ, có thể trả về ít hơn k kết quả

---

### 2.5. Tiêu chí 5: Thứ tự từ KNN (Ranking)

**Vị trí**: Dòng 552, 586
```python
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    # ...
    results.append({
        'rank': len(results) + 1,
        # ...
    })
```

**Mô tả**:
- KNN trả về neighbors đã được **sắp xếp theo distance tăng dần**
- Rank được gán theo thứ tự xuất hiện trong kết quả KNN
- Rank 1 = Giống nhất (distance nhỏ nhất)

**Thứ tự**:
1. Rank 1: Distance nhỏ nhất → Similarity cao nhất
2. Rank 2: Distance nhỏ thứ 2 → Similarity cao thứ 2
3. ...
4. Rank K: Distance nhỏ thứ K → Similarity cao thứ K

**Ưu tiên**: **CAO** - Xác định thứ tự ban đầu

**Lưu ý**: 
- Sau khi boost, thứ tự có thể thay đổi nếu có nhiều mẫu cùng base_similarity
- Nhưng thực tế, KNN đã sắp xếp tốt nên ít khi xảy ra

---

## 3. QUY TRÌNH ĐÁNH GIÁ VÀ SẮP XẾP

### 3.1. Bước 1: Tìm K-Nearest Neighbors

```
Input: Feature vector của audio input
↓
KNN Query với metric='cosine'
↓
Output: search_k neighbors đã sắp xếp theo distance tăng dần
```

**Tiêu chí**: Cosine Distance (nhỏ nhất = giống nhất)

---

### 3.2. Bước 2: Tính Base Similarity

```
For mỗi neighbor:
    distance = cosine_distance từ KNN
    ↓
    base_similarity = (1 - distance) × 100
```

**Tiêu chí**: Base Similarity Score (cao nhất = giống nhất)

---

### 3.3. Bước 3: Áp dụng Boost (nếu có)

```
For mỗi neighbor:
    if (cùng_dialect AND boost_enabled):
        similarity = min(100, base_similarity + 20)
    else:
        similarity = base_similarity
```

**Tiêu chí**: Boost Similarity (tăng ưu tiên cho cùng dialect)

---

### 3.4. Bước 4: Filter theo Vùng miền

```
For mỗi neighbor:
    if (filter_dialect AND sample_dialect != filter_dialect):
        continue  # Bỏ qua
    else:
        thêm vào results
```

**Tiêu chí**: Filter Dialect (chỉ giữ cùng vùng miền)

---

### 3.5. Bước 5: Gán Rank và Trả về

```
For mỗi neighbor (sau filter):
    rank = len(results) + 1
    thêm vào results với rank, similarity, ...
    ↓
    if len(results) >= k:
        break  # Đủ kết quả
```

**Tiêu chí**: Thứ tự từ KNN (giữ nguyên ranking)

---

## 4. BẢNG TÓM TẮT CÁC TIÊU CHÍ

| Tiêu chí | Vị trí | Công thức | Ưu tiên | Ảnh hưởng |
|----------|--------|-----------|---------|-----------|
| **Cosine Distance** | Dòng 548 | `1 - cosine_similarity` | **CAO NHẤT** | Xác định thứ tự ban đầu |
| **Base Similarity** | Dòng 554 | `(1 - dist) × 100` | **CAO** | Chuyển đổi để hiển thị |
| **Boost Similarity** | Dòng 560-563 | `min(100, base + 20)` | **TRUNG BÌNH** | Điều chỉnh cho cùng dialect |
| **Filter Dialect** | Dòng 565-567 | `dialect == filter_dialect` | **CAO** | Lọc kết quả |
| **Ranking** | Dòng 552, 586 | Thứ tự từ KNN | **CAO** | Xác định top K |

---

## 5. VÍ DỤ CỤ THỂ

### Ví dụ 1: Không filter, không boost (k=5)

**Input**: Audio với features vector

**KNN trả về** (đã sắp xếp):
1. Sample A: distance=0.15 → base_similarity=85%
2. Sample B: distance=0.20 → base_similarity=80%
3. Sample C: distance=0.25 → base_similarity=75%
4. Sample D: distance=0.30 → base_similarity=70%
5. Sample E: distance=0.35 → base_similarity=65%

**Kết quả** (giữ nguyên thứ tự):
- Rank 1: Sample A (85%)
- Rank 2: Sample B (80%)
- Rank 3: Sample C (75%)
- Rank 4: Sample D (70%)
- Rank 5: Sample E (65%)

**Tiêu chí**: Cosine Distance → Base Similarity

---

### Ví dụ 2: Có boost, không filter (k=5)

**Input**: Audio Bắc, boost enabled

**KNN trả về**:
1. Sample A (Bắc): distance=0.20 → base=80% → **final=100%** (boost)
2. Sample B (Trung): distance=0.15 → base=85% → final=85% (không boost)
3. Sample C (Bắc): distance=0.25 → base=75% → **final=95%** (boost)
4. Sample D (Nam): distance=0.18 → base=82% → final=82% (không boost)
5. Sample E (Bắc): distance=0.30 → base=70% → **final=90%** (boost)

**Kết quả** (sau boost, có thể thay đổi thứ tự):
- Rank 1: Sample A (100%) - Bắc, boost
- Rank 2: Sample B (85%) - Trung, không boost
- Rank 3: Sample C (95%) - Bắc, boost
- Rank 4: Sample D (82%) - Nam, không boost
- Rank 5: Sample E (90%) - Bắc, boost

**Tiêu chí**: Cosine Distance → Base Similarity → Boost Similarity

---

### Ví dụ 3: Có filter, có boost (k=5, filter=Bắc)

**Input**: Audio Bắc, filter=Bắc, boost enabled

**KNN trả về** (tìm 15 neighbors = k×3):
1. Sample A (Bắc): distance=0.20 → base=80% → final=100% (boost) ✅
2. Sample B (Trung): distance=0.15 → base=85% → **BỎ QUA** (khác dialect)
3. Sample C (Bắc): distance=0.25 → base=75% → final=95% (boost) ✅
4. Sample D (Nam): distance=0.18 → base=82% → **BỎ QUA** (khác dialect)
5. Sample E (Bắc): distance=0.30 → base=70% → final=90% (boost) ✅
6. Sample F (Bắc): distance=0.35 → base=65% → final=85% (boost) ✅
7. Sample G (Trung): distance=0.22 → base=78% → **BỎ QUA** (khác dialect)
8. Sample H (Bắc): distance=0.40 → base=60% → final=80% (boost) ✅
9. ...

**Kết quả** (sau filter, lấy 5 đầu tiên):
- Rank 1: Sample A (100%)
- Rank 2: Sample C (95%)
- Rank 3: Sample E (90%)
- Rank 4: Sample F (85%)
- Rank 5: Sample H (80%)

**Tiêu chí**: Cosine Distance → Base Similarity → Filter → Boost → Ranking

---

## 6. CÁC YẾU TỐ ẢNH HƯỞNG ĐẾN KẾT QUẢ

### 6.1. Yếu tố chính

1. **Cosine Distance** (Quan trọng nhất)
   - Phụ thuộc vào độ tương tự của feature vectors
   - Được tính từ 63 features đã chuẩn hóa
   - Quyết định thứ tự ban đầu

2. **Vùng miền (Dialect)**
   - Ảnh hưởng đến boost (+20%)
   - Ảnh hưởng đến filter (bỏ qua nếu khác)
   - Có thể thay đổi ranking

3. **Số lượng neighbors tìm (search_k)**
   - Không filter: `search_k = k`
   - Có filter: `search_k = k × 3`
   - Ảnh hưởng đến số lượng mẫu được xem xét

### 6.2. Yếu tố phụ

1. **Boost enabled/disabled**
   - Bật: Tăng +20% cho cùng dialect
   - Tắt: Giữ nguyên base similarity

2. **Filter dialect**
   - Có: Chỉ giữ cùng vùng miền
   - Không: Giữ tất cả

3. **Chất lượng features**
   - Features tốt → Distance chính xác hơn
   - Features kém → Distance không chính xác

---

## 7. KẾT LUẬN

### Tiêu chí chính để đánh giá top K:

1. **Cosine Distance** (100% quyết định)
   - Tiêu chí quan trọng nhất
   - Xác định thứ tự ban đầu
   - Tất cả tính toán dựa trên nó

2. **Base Similarity** (100% phụ thuộc Cosine)
   - Chỉ là cách biểu diễn
   - Giữ nguyên thứ tự

3. **Boost Similarity** (~20% điều chỉnh)
   - Chỉ điều chỉnh, không thay đổi cơ bản
   - Có thể thay đổi ranking nếu có nhiều mẫu cùng base

4. **Filter Dialect** (Quyết định mẫu nào được giữ)
   - Lọc kết quả
   - Không ảnh hưởng đến thứ tự (chỉ bỏ qua)

5. **Ranking** (Thứ tự từ KNN)
   - Giữ nguyên thứ tự từ KNN
   - Rank 1 = Giống nhất

### Công thức tổng quát:

```
1. Cosine Distance = 1 - Cosine Similarity (từ KNN)
2. Base Similarity = (1 - Distance) × 100
3. Final Similarity = Base + Boost (nếu cùng dialect)
4. Filter: Chỉ giữ nếu cùng dialect (nếu có filter)
5. Rank: Theo thứ tự từ KNN (distance tăng dần)
```

### Kết luận:

**Tiêu chí chủ yếu**: **Cosine Distance** - Quyết định 100% thứ tự ranking ban đầu

**Tiêu chí phụ**: Boost và Filter - Chỉ điều chỉnh và lọc, không thay đổi cơ bản

