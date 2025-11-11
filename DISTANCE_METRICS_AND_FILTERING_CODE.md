# Vị trí Code: Đo khoảng cách và Lọc/Tối ưu kết quả

## File: `voice_search_gui_enhanced.py`

---

## 2.4.2. Các phương pháp đo khoảng cách

### 1. Khởi tạo KNN Model với Metric - Dòng 144

**Code**:
```python
self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
```

**Giải thích**:
- `metric='cosine'`: Sử dụng **Cosine Similarity** (mặc định)
- Có thể thay đổi thành:
  - `metric='euclidean'` → **Euclidean Distance**
  - `metric='manhattan'` → **Manhattan Distance**
  - `metric='correlation'` → **Correlation Distance**

**Vị trí**: Trong hàm `train_model()`, dòng 144

---

### 2. Tính toán Distance và Similarity - Dòng 548, 554

**Code tìm KNN**:
```python
# Tìm K nearest neighbors
distances, indices = self.knn_model.kneighbors(feature_vector_scaled, n_neighbors=min(search_k, len(self.df_train)))
```

**Code chuyển đổi distance → similarity**:
```python
base_similarity = (1 - dist) * 100
```

**Giải thích**:
- Dòng 548: `kneighbors()` trả về `distances` và `indices`
- Dòng 554: Chuyển đổi distance thành similarity score (0-100%)
- Công thức: `similarity = (1 - distance) × 100`

**Vị trí**: Trong hàm `search_similar_voices()`, dòng 548 và 554

---

### 3. Các Metric có thể sử dụng

**Trong scikit-learn NearestNeighbors**, các metric được hỗ trợ:

| Metric | Tên trong code | Công thức |
|--------|----------------|-----------|
| **Cosine Similarity** | `'cosine'` | `cos(θ) = (A·B)/(\|A\|\|B\|)` |
| **Euclidean Distance** | `'euclidean'` | `√(Σ(Ai-Bi)²)` |
| **Manhattan Distance** | `'manhattan'` | `Σ\|Ai-Bi\|` |
| **Correlation Distance** | `'correlation'` | `1 - correlation_coefficient` |

**Để thay đổi metric**, sửa dòng 144:
```python
# Cosine (hiện tại)
self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')

# Hoặc đổi thành:
self.knn_model = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='brute')
self.knn_model = NearestNeighbors(n_neighbors=10, metric='manhattan', algorithm='brute')
self.knn_model = NearestNeighbors(n_neighbors=10, metric='correlation', algorithm='brute')
```

---

## 2.4.3. Lọc và tối ưu kết quả

### 1. Lọc theo vùng miền - Dòng 545-567

**Code**:
```python
# Tìm nhiều neighbors hơn nếu cần filter
search_k = k * 3 if filter_dialect else k

# Tìm K nearest neighbors
distances, indices = self.knn_model.kneighbors(feature_vector_scaled, n_neighbors=min(search_k, len(self.df_train)))

# Lấy thông tin các samples tương tự
results = []
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    sample = self.df_train.iloc[idx]
    base_similarity = (1 - dist) * 100
    
    sample_dialect = sample.get('dialect', 'N/A')
    
    # ... boost similarity ...
    
    # Filter theo dialect nếu được yêu cầu
    if filter_dialect and sample_dialect != filter_dialect:
        continue  # ← Bỏ qua nếu không cùng vùng miền
    
    # ... thêm vào results ...
```

**Giải thích**:
- Dòng 545: Tìm nhiều neighbors hơn (k×3) nếu cần filter để đảm bảo đủ kết quả
- Dòng 566-567: Bỏ qua các mẫu không cùng vùng miền nếu `filter_dialect` được set

**Vị trí**: Trong hàm `search_similar_voices()`, dòng 545-567

---

### 2. Boost cùng vùng miền - Dòng 558-563

**Code**:
```python
# Boost similarity nếu cùng dialect
similarity = base_similarity
if boost_same_dialect and filter_dialect and sample_dialect == filter_dialect:
    similarity = min(100, base_similarity + 20)  # Boost +20%
elif boost_same_dialect and self.input_dialect and sample_dialect == self.input_dialect:
    similarity = min(100, base_similarity + 20)  # Boost +20%
```

**Giải thích**:
- Dòng 559: Khởi tạo similarity = base_similarity
- Dòng 560-561: Nếu bật boost và có filter dialect và cùng dialect → tăng +20%
- Dòng 562-563: Nếu bật boost và input có dialect và cùng dialect → tăng +20%
- `min(100, ...)`: Đảm bảo similarity không vượt quá 100%

**Vị trí**: Trong hàm `search_similar_voices()`, dòng 558-563

---

### 3. Sắp xếp kết quả - Dòng 585-599

**Code tạo kết quả**:
```python
results.append({
    'rank': len(results) + 1,
    'audio_name': sample['audio_name'],
    'speaker_id': speaker_id,
    'speaker_name': speaker_name,
    'similarity': round(similarity, 2),
    'base_similarity': round(base_similarity, 2),  # Similarity gốc
    'distance': float(dist),
    'dialect': sample_dialect,
    'features': sample_features  # Thêm features để so sánh
})

# Dừng khi đủ kết quả
if len(results) >= k:
    break
```

**Giải thích**:
- Dòng 586: Rank được tính tự động (1, 2, 3, ...)
- Dòng 590: Similarity đã được boost (nếu có)
- Dòng 591: Base similarity gốc (trước khi boost)
- Kết quả đã được sắp xếp tự động vì KNN trả về theo thứ tự distance tăng dần

**Vị trí**: Trong hàm `search_similar_voices()`, dòng 585-599

---

### 4. Hiển thị kết quả trong GUI - Dòng 1403-1418

**Code hiển thị**:
```python
# Hiển thị kết quả
for result in results:
    similarity = result['similarity']
    color_tag = 'high' if similarity > 80 else 'medium' if similarity > 60 else 'low'
    
    # Tạo link text với icon folder
    audio_name = result['audio_name']
    link_text = f"📁 Mở folder"
    
    self.results_tree.insert('', END, values=(
        result['rank'],              # Rank
        f"{similarity:.2f}%",        # Similarity
        result['speaker_name'],       # Speaker (tên người nói)
        result['audio_name'],         # Audio Name (file audio)
        result['dialect'],            # Dialect (vùng miền)
        link_text                    # Link (để phát audio)
    ), tags=(color_tag, 'link'))
```

**Giải thích**:
- Dòng 1404-1405: Xác định màu sắc dựa trên similarity:
  - `> 80%`: màu xanh lá (high)
  - `> 60%`: màu cam (medium)
  - `≤ 60%`: màu đỏ (low)
- Dòng 1411-1418: Chèn vào TreeView với các cột:
  - Rank
  - Similarity (%)
  - Speaker (tên người nói)
  - Audio Name
  - Dialect
  - Link

**Vị trí**: Trong hàm `on_search_complete()`, dòng 1403-1418

---

### 5. Cấu hình TreeView Columns - Dòng 785-799

**Code định nghĩa cột**:
```python
columns = ('Rank', 'Similarity', 'Speaker', 'Audio Name', 'Dialect', 'Link')
self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=12)

# Cấu hình cột
for col in columns:
    self.results_tree.heading(col, text=col)
    if col == 'Rank':
        self.results_tree.column(col, width=60, anchor=CENTER)
    elif col == 'Similarity':
        self.results_tree.column(col, width=110, anchor=CENTER)
    elif col == 'Speaker':
        self.results_tree.column(col, width=140)
    elif col == 'Dialect':
        self.results_tree.column(col, width=90, anchor=CENTER)
    # ...
```

**Vị trí**: Trong hàm `create_widgets()`, dòng 785-799

---

### 6. Gọi hàm search với filter và boost - Dòng 1369-1382

**Code gọi hàm search**:
```python
k = self.k_value.get()

# Lấy filter dialect
filter_dialect = None
selected_dialect = self.dialect_var.get()
if selected_dialect != "Tất cả":
    filter_dialect = selected_dialect

# Boost cùng dialect
boost_same = self.boost_same_dialect.get()

results, input_features = self.voice_engine.search_similar_voices(
    audio_file, k, progress_callback, 
    filter_dialect=filter_dialect,
    boost_same_dialect=boost_same
)
```

**Giải thích**:
- Dòng 1370-1373: Lấy filter dialect từ dropdown (nếu không phải "Tất cả")
- Dòng 1376: Lấy giá trị checkbox "Boost cùng vùng miền"
- Dòng 1378-1382: Gọi hàm `search_similar_voices()` với các tham số

**Vị trí**: Trong hàm `search_voices()`, dòng 1369-1382

---

## Tóm tắt vị trí code:

| Chức năng | Dòng code | Mô tả |
|-----------|-----------|-------|
| **Khởi tạo KNN với metric** | 144 | `NearestNeighbors(metric='cosine')` |
| **Tìm KNN** | 548 | `kneighbors()` trả về distances và indices |
| **Chuyển distance → similarity** | 554 | `(1 - dist) * 100` |
| **Tăng search_k khi filter** | 545 | `k * 3 if filter_dialect else k` |
| **Lọc theo dialect** | 566-567 | `if filter_dialect and sample_dialect != filter_dialect: continue` |
| **Boost similarity** | 560-563 | `similarity = min(100, base_similarity + 20)` |
| **Tạo kết quả với rank** | 585-595 | Dictionary chứa rank, similarity, speaker, audio_name, dialect |
| **Hiển thị trong TreeView** | 1403-1418 | Insert vào TreeView với màu sắc theo similarity |
| **Gọi search với filter** | 1369-1382 | Lấy filter và boost từ GUI, gọi `search_similar_voices()` |

---

## Code đầy đủ để chụp:

### 1. Khởi tạo KNN Model (Dòng 144):
```python
self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
```

### 2. Tìm kiếm và Lọc/Boost (Dòng 545-599):
```python
# Tìm nhiều neighbors hơn nếu cần filter
search_k = k * 3 if filter_dialect else k

# Tìm K nearest neighbors
distances, indices = self.knn_model.kneighbors(feature_vector_scaled, n_neighbors=min(search_k, len(self.df_train)))

# Lấy thông tin các samples tương tự
results = []
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    sample = self.df_train.iloc[idx]
    base_similarity = (1 - dist) * 100
    
    sample_dialect = sample.get('dialect', 'N/A')
    
    # Boost similarity nếu cùng dialect
    similarity = base_similarity
    if boost_same_dialect and filter_dialect and sample_dialect == filter_dialect:
        similarity = min(100, base_similarity + 20)  # Boost +20%
    elif boost_same_dialect and self.input_dialect and sample_dialect == self.input_dialect:
        similarity = min(100, base_similarity + 20)  # Boost +20%
    
    # Filter theo dialect nếu được yêu cầu
    if filter_dialect and sample_dialect != filter_dialect:
        continue
    
    # ... lấy thông tin speaker ...
    
    results.append({
        'rank': len(results) + 1,
        'audio_name': sample['audio_name'],
        'speaker_id': speaker_id,
        'speaker_name': speaker_name,
        'similarity': round(similarity, 2),
        'base_similarity': round(base_similarity, 2),
        'distance': float(dist),
        'dialect': sample_dialect,
        'features': sample_features
    })
    
    # Dừng khi đủ kết quả
    if len(results) >= k:
        break
```

### 3. Hiển thị kết quả (Dòng 1403-1418):
```python
for result in results:
    similarity = result['similarity']
    color_tag = 'high' if similarity > 80 else 'medium' if similarity > 60 else 'low'
    
    link_text = f"📁 Mở folder"
    
    self.results_tree.insert('', END, values=(
        result['rank'],
        f"{similarity:.2f}%",
        result['speaker_name'],
        result['audio_name'],
        result['dialect'],
        link_text
    ), tags=(color_tag, 'link'))
```

