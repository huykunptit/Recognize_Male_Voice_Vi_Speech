# Các hàm Load dữ liệu training - Vị trí và Code

## File: `voice_search_gui_enhanced.py`

### 1. Hàm `load_training_data()` - Dòng 98-114

**Chức năng**: Đọc tất cả file CSV trong thư mục `super_metadata/male_only/` và gộp thành một DataFrame

```python
def load_training_data(self):
    """Load và merge tất cả CSV files từ super_metadata/male_only"""
    all_data = []
    csv_files = sorted(Path(self.super_metadata_folder).glob("*.csv"))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            all_data.append(df)
        except Exception as e:
            print(f"Lỗi khi load {csv_file.name}: {e}")
    
    if not all_data:
        raise ValueError("Không tìm thấy dữ liệu training!")
    
    self.df_train = pd.concat(all_data, ignore_index=True)
    return True
```

**Giải thích**:
- Dòng 100: Tạo list `all_data` để chứa các DataFrame
- Dòng 101: Tìm tất cả file CSV trong thư mục `super_metadata/male_only/` và sắp xếp
- Dòng 103-108: Đọc từng file CSV và thêm vào list (có xử lý lỗi)
- Dòng 110-111: Kiểm tra nếu không có dữ liệu thì báo lỗi
- Dòng 113: Gộp tất cả DataFrame thành một DataFrame duy nhất bằng `pd.concat()`

---

### 2. Hàm `get_feature_columns()` - Dòng 116-122

**Chức năng**: Loại bỏ các cột không phải feature (audio_name, dialect, gender, speaker)

```python
def get_feature_columns(self):
    """Lấy danh sách các cột features"""
    if self.feature_columns is None:
        exclude_cols = ['audio_name', 'dialect', 'gender', 'speaker']
        self.feature_columns = [col for col in self.df_train.columns 
                               if col not in exclude_cols]
    return self.feature_columns
```

**Giải thích**:
- Dòng 119: Định nghĩa các cột cần loại bỏ (metadata, không phải feature)
- Dòng 120-121: Lọc các cột, chỉ giữ lại các cột không nằm trong `exclude_cols`
- Dòng 122: Trả về danh sách các cột feature

---

### 3. Hàm `train_model()` - Dòng 124-168 (Phần liên quan)

**Chức năng**: Sử dụng `load_training_data()` và `get_feature_columns()` để train model

**Phần code liên quan đến load dữ liệu**:

```python
def train_model(self, progress_callback=None):
    """Train KNN model từ dữ liệu training"""
    if progress_callback:
        progress_callback("Đang load dữ liệu training...")
    
    self.load_training_data()  # ← Gọi hàm load dữ liệu
    self.load_speaker_database()
    
    feature_cols = self.get_feature_columns()  # ← Lấy danh sách feature columns
    
    if progress_callback:
        progress_callback("Đang chuẩn hóa dữ liệu...")
    
    X_train = self.df_train[feature_cols].fillna(0).values  # ← Chỉ lấy các cột feature
    self.scaler = StandardScaler()
    X_train_scaled = self.scaler.fit_transform(X_train)
    
    # ... phần train KNN model ...
```

**Giải thích**:
- Dòng 129: Gọi `load_training_data()` để load và gộp tất cả CSV
- Dòng 132: Gọi `get_feature_columns()` để lấy danh sách các cột feature (đã loại bỏ metadata)
- Dòng 137: Chỉ lấy các cột feature từ DataFrame, thay NaN bằng 0

---

## Tóm tắt quy trình Load dữ liệu training:

1. **`load_training_data()`** (dòng 98-114):
   - ✅ Đọc tất cả file CSV trong `super_metadata/male_only/`
   - ✅ Gộp các DataFrame thành `self.df_train`

2. **`get_feature_columns()`** (dòng 116-122):
   - ✅ Loại bỏ các cột: `audio_name`, `dialect`, `gender`, `speaker`
   - ✅ Trả về danh sách các cột feature

3. **Trong `train_model()`** (dòng 137):
   - ✅ Sử dụng `feature_cols` để chỉ lấy các cột feature từ `self.df_train`

---

## Các file khác có cùng logic:

- `voice_search_gui.py` - dòng 98-114 (giống hệt)
- `voice_search_cli.py` - dòng 85-101 (giống hệt)
- `voice_search_app.py` - dòng 84-100 (giống hệt)

**Lưu ý**: Tất cả các file đều có cùng logic, bạn có thể chụp từ file `voice_search_gui_enhanced.py` vì đây là file chính và được cập nhật mới nhất.

---

## Code đầy đủ để chụp (3 hàm chính):

### Hàm 1: load_training_data()
```python
def load_training_data(self):
    """Load và merge tất cả CSV files từ super_metadata/male_only"""
    all_data = []
    csv_files = sorted(Path(self.super_metadata_folder).glob("*.csv"))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            all_data.append(df)
        except Exception as e:
            print(f"Lỗi khi load {csv_file.name}: {e}")
    
    if not all_data:
        raise ValueError("Không tìm thấy dữ liệu training!")
    
    self.df_train = pd.concat(all_data, ignore_index=True)
    return True
```

### Hàm 2: get_feature_columns()
```python
def get_feature_columns(self):
    """Lấy danh sách các cột features"""
    if self.feature_columns is None:
        exclude_cols = ['audio_name', 'dialect', 'gender', 'speaker']
        self.feature_columns = [col for col in self.df_train.columns 
                               if col not in exclude_cols]
    return self.feature_columns
```

### Hàm 3: Phần sử dụng trong train_model()
```python
# Trong hàm train_model(), dòng 129-137
self.load_training_data()  # Load và gộp CSV
self.load_speaker_database()

feature_cols = self.get_feature_columns()  # Loại bỏ cột metadata

X_train = self.df_train[feature_cols].fillna(0).values  # Chỉ lấy features
```

