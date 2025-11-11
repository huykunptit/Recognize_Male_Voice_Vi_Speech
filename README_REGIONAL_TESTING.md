# Hướng Dẫn Test Các Thuật Toán Phát Hiện Vùng Miền

## Tổng quan

Bộ script này giúp bạn test và so sánh nhiều thuật toán machine learning khác nhau cho việc phát hiện vùng miền (North/Central/South) từ đặc trưng giọng nói.

## Danh sách các file test

### 1. `test_regional_classifiers.py` - **TEST TẤT CẢ**
Script chính để test và so sánh tất cả các thuật toán:
- Random Forest
- XGBoost
- LightGBM
- SVM (RBF, Linear, Poly)
- Neural Network (MLP)
- K-NN
- Logistic Regression
- Decision Tree
- Naive Bayes
- AdaBoost
- Gradient Boosting
- Ensemble Voting

**Chạy:** `python test_regional_classifiers.py`

### 2. `test_xgboost_regional.py` - XGBoost
Test XGBoost với nhiều config khác nhau (n_estimators, max_depth, learning_rate)

**Chạy:** `python test_xgboost_regional.py`

### 3. `test_lightgbm_regional.py` - LightGBM
Test LightGBM với nhiều config khác nhau

**Chạy:** `python test_lightgbm_regional.py`

### 4. `test_svm_regional.py` - Support Vector Machine
Test SVM với nhiều kernel (RBF, Linear, Poly, Sigmoid) và parameters

**Chạy:** `python test_svm_regional.py`

### 5. `test_neural_network_regional.py` - Neural Network (MLP)
Test Multi-Layer Perceptron với nhiều architecture khác nhau

**Chạy:** `python test_neural_network_regional.py`

### 6. `test_ensemble_regional.py` - Ensemble Methods
Test các phương pháp ensemble (Voting, AdaBoost, Gradient Boosting)

**Chạy:** `python test_ensemble_regional.py`

### 7. `test_knn_regional.py` - K-Nearest Neighbors
Test K-NN với nhiều k và metric (Euclidean, Manhattan, Cosine)

**Chạy:** `python test_knn_regional.py`

### 8. `test_random_forest_variants.py` - Random Forest Variants
Test Random Forest với nhiều biến thể (n_estimators, max_depth, min_samples_split)

**Chạy:** `python test_random_forest_variants.py`

## Cài đặt dependencies

Trước khi chạy, cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Các thư viện mới cần thêm:
- `xgboost` - XGBoost classifier
- `lightgbm` - LightGBM classifier  
- `joblib` - Lưu/load models

## Kết quả

Mỗi script sẽ:
1. Load dữ liệu từ `super_metadata/male_only_merged.csv`
2. Split data thành train/test (80/20)
3. Test nhiều config khác nhau
4. In ra:
   - Test Accuracy
   - Cross-validation scores
   - Confusion Matrix
   - Comparison table
5. Lưu model tốt nhất vào file `.joblib`

## Model files được tạo

- `best_regional_classifier.joblib` - Model tốt nhất từ test tất cả
- `xgboost_regional_classifier.joblib` - Model XGBoost tốt nhất
- `lightgbm_regional_classifier.joblib` - Model LightGBM tốt nhất
- `svm_regional_classifier.joblib` - Model SVM tốt nhất
- `mlp_regional_classifier.joblib` - Model Neural Network tốt nhất
- `ensemble_regional_classifier.joblib` - Model Ensemble tốt nhất
- `knn_regional_classifier.joblib` - Model K-NN tốt nhất
- `rf_regional_classifier.joblib` - Model Random Forest tốt nhất

## Khuyến nghị

1. **Chạy test tất cả trước:** `python test_regional_classifiers.py`
   - Sẽ cho bạn cái nhìn tổng quan về thuật toán nào tốt nhất

2. **Chạy các test riêng lẻ** để tìm config tối ưu cho từng thuật toán

3. **So sánh kết quả** và chọn thuật toán tốt nhất

4. **Integrate model tốt nhất** vào `voice_desktop_app_final.py`

## Thời gian chạy

- `test_regional_classifiers.py`: ~10-15 phút (test tất cả)
- Các test riêng lẻ: ~2-5 phút mỗi cái

## Lưu ý

- Đảm bảo file `super_metadata/male_only_merged.csv` đã có dữ liệu được training (không phải toàn 0)
- Nếu chưa training, chạy `python run_training.py` trước
- Kết quả có thể khác nhau tùy vào dữ liệu và random seed

