# Vị trí Code Trích xuất Đặc trưng Audio

## File: `voice_search_gui_enhanced.py`

### Hàm chính: `extract_audio_features()` - Dòng 239-480

---

## 1. PITCH (Tần số cơ bản) - Dòng 425-454

**Đặc trưng**: `pitch_mean`, `pitch_std`, `pitch_range`

**Code**:
```python
# Pitch - try pyin, fallback to yin, fallback to zeros
try:
    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    if f0 is None:
        raise Exception("pyin returned None")
    f0_clean = f0[~np.isnan(f0)]
    if f0_clean.size > 0:
        features['pitch_mean'] = float(np.mean(f0_clean))
        features['pitch_std'] = float(np.std(f0_clean))
        features['pitch_range'] = float(np.max(f0_clean) - np.min(f0_clean))
    else:
        features['pitch_mean'] = defaults['pitch_mean']
        features['pitch_std'] = defaults['pitch_std']
        features['pitch_range'] = defaults['pitch_range']
except Exception:
    try:
        f0_yin = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0_yin[~np.isnan(f0_yin)]
        if f0_clean.size > 0:
            features['pitch_mean'] = float(np.mean(f0_clean))
            features['pitch_std'] = float(np.std(f0_clean))
            features['pitch_range'] = float(np.max(f0_clean) - np.min(f0_clean))
        else:
            features['pitch_mean'] = defaults['pitch_mean']
            features['pitch_std'] = defaults['pitch_std']
            features['pitch_range'] = defaults['pitch_range']
    except Exception:
        features['pitch_mean'] = defaults['pitch_mean']
        features['pitch_std'] = defaults['pitch_std']
        features['pitch_range'] = defaults['pitch_range']
```

**Giải thích**:
- Dòng 427: Sử dụng thuật toán **PYIN** (Probabilistic YIN) - phương pháp chính
- Dòng 441: Fallback sang thuật toán **YIN** nếu PYIN thất bại
- Tính mean, std, và range từ f0 (fundamental frequency)

---

## 2. MFCC (Mel-Frequency Cepstral Coefficients) - Dòng 322-331

**Đặc trưng**: 13 coefficients × 2 (mean, std) = 26 features

**Code**:
```python
# MFCCs (13)
try:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
except Exception:
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = defaults[f'mfcc_{i+1}_mean']
        features[f'mfcc_{i+1}_std'] = defaults[f'mfcc_{i+1}_std']
```

**Giải thích**:
- Dòng 324: Trích xuất 13 MFCC coefficients từ phổ Mel
- Dòng 325-327: Tính mean và std cho mỗi coefficient
- Tổng cộng: `mfcc_1_mean` đến `mfcc_13_std` = 26 features

---

## 3. SPECTRAL (Phổ tần số) - Nhiều đoạn code

### 3.1. Spectral Centroid - Dòng 295-302
```python
# Spectral Centroid
try:
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(sc))
    features['spectral_centroid_std'] = float(np.std(sc))
except Exception:
    features['spectral_centroid_mean'] = defaults['spectral_centroid_mean']
    features['spectral_centroid_std'] = defaults['spectral_centroid_std']
```

### 3.2. Spectral Rolloff - Dòng 304-311
```python
# Spectral Rolloff
try:
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(roll))
    features['spectral_rolloff_std'] = float(np.std(roll))
except Exception:
    features['spectral_rolloff_mean'] = defaults['spectral_rolloff_mean']
    features['spectral_rolloff_std'] = defaults['spectral_rolloff_std']
```

### 3.3. Spectral Bandwidth - Dòng 377-383
```python
# Spectral Bandwidth
try:
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = float(np.mean(bw))
    features['spectral_bandwidth_std'] = float(np.std(bw))
except Exception:
    features['spectral_bandwidth_mean'] = defaults['spectral_bandwidth_mean']
    features['spectral_bandwidth_std'] = defaults['spectral_bandwidth_std']
```

### 3.4. Spectral Flatness - Dòng 385-391
```python
try:
    flat = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = float(np.mean(flat))
    features['spectral_flatness_std'] = float(np.std(flat))
except Exception:
    features['spectral_flatness_mean'] = defaults['spectral_flatness_mean']
    features['spectral_flatness_std'] = defaults['spectral_flatness_std']
```

### 3.5. Spectral Contrast - Dòng 342-349
```python
# Spectral Contrast
try:
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = float(np.mean(contrast))
    features['spectral_contrast_std'] = float(np.std(contrast))
except Exception:
    features['spectral_contrast_mean'] = defaults['spectral_contrast_mean']
    features['spectral_contrast_std'] = defaults['spectral_contrast_std']
```

### 3.6. Spectral Slope - Dòng 393-400
```python
# Spectral Slope - Dùng hàm compute_spectral_slope
try:
    slopes = self.compute_spectral_slope(y, sr)
    features['spectral_slope_mean'] = float(np.mean(slopes))
    features['spectral_slope_std'] = float(np.std(slopes))
except Exception:
    features['spectral_slope_mean'] = defaults['spectral_slope_mean']
    features['spectral_slope_std'] = defaults['spectral_slope_std']
```

**Hàm hỗ trợ `compute_spectral_slope()` - Dòng 187-209**:
```python
def compute_spectral_slope(self, y, sr, n_fft=2048, hop_length=512):
    """
    Compute spectral slope per frame by fitting a line to log-magnitude vs frequency.
    Returns array of slopes (one per frame).
    """
    try:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    except Exception:
        return np.zeros(1, dtype=float)
    freqs = np.linspace(0, sr / 2, S.shape[0])
    slopes = []
    for i in range(S.shape[1]):
        mag = S[:, i]
        if np.all(mag == 0):
            slopes.append(0.0)
            continue
        mag_log = np.log1p(mag)
        try:
            a, _ = np.polyfit(freqs, mag_log, 1)
        except Exception:
            a = 0.0
        slopes.append(float(a))
    return np.array(slopes)
```

### 3.7. Spectral Kurtosis & Skewness - Dòng 402-415
```python
# Spectral Kurtosis & Skewness - Từ STFT
try:
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spec_kurt = kurtosis(S, axis=0, fisher=True, nan_policy='omit')
    spec_skew = skew(S, axis=0, nan_policy='omit')
    features['spectral_kurtosis_mean'] = float(np.nanmean(spec_kurt))
    features['spectral_kurtosis_std'] = float(np.nanstd(spec_kurt))
    features['spectral_skewness_mean'] = float(np.nanmean(spec_skew))
    features['spectral_skewness_std'] = float(np.nanstd(spec_skew))
except Exception:
    features['spectral_kurtosis_mean'] = defaults['spectral_kurtosis_mean']
    features['spectral_kurtosis_std'] = defaults['spectral_kurtosis_std']
    features['spectral_skewness_mean'] = defaults['spectral_skewness_mean']
    features['spectral_skewness_std'] = defaults['spectral_skewness_std']
```

### 3.8. Spectral Flux - Dòng 417-423
```python
# Spectral Flux - Tính từ STFT diff
try:
    S = np.abs(librosa.stft(y))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    features['spectral_flux'] = float(np.mean(flux)) if flux.size > 0 else 0.0
except Exception:
    features['spectral_flux'] = defaults['spectral_flux']
```

---

## 4. TEMPORAL (Thời gian)

### 4.1. ZCR (Zero Crossing Rate) - Dòng 313-320
```python
# ZCR
try:
    z = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(z))
    features['zcr_std'] = float(np.std(z))
except Exception:
    features['zcr_mean'] = defaults['zcr_mean']
    features['zcr_std'] = defaults['zcr_std']
```

### 4.2. RMS (Root Mean Square) - Dòng 285-293
```python
# RMS
try:
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    features['rms_max'] = float(np.max(rms))
    features['rms_min'] = float(np.min(rms))
except Exception:
    features.update({k: defaults[k] for k in ['rms_mean','rms_std','rms_max','rms_min']})
```

### 4.3. Tempo - Dòng 360-365
```python
# Tempo
try:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)
except Exception:
    features['tempo'] = defaults['tempo']
```

### 4.4. Duration - Dòng 281-283
```python
# Duration
duration = float(len(y) / float(sr))
features['duration'] = duration
```

### 4.5. Onset Strength - Dòng 367-374
```python
# Onset Strength
try:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features['onset_strength_mean'] = float(np.mean(onset_env))
    features['onset_strength_std'] = float(np.std(onset_env))
except Exception:
    features['onset_strength_mean'] = defaults['onset_strength_mean']
    features['onset_strength_std'] = defaults['onset_strength_std']
```

---

## 5. HARMONIC (Hài hòa)

### 5.1. Chroma - Dòng 333-340
```python
# Chroma
try:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma))
    features['chroma_std'] = float(np.std(chroma))
except Exception:
    features['chroma_mean'] = defaults['chroma_mean']
    features['chroma_std'] = defaults['chroma_std']
```

### 5.2. Tonnetz - Dòng 351-358
```python
# Tonnetz
try:
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features['tonnetz_mean'] = float(np.mean(tonnetz))
    features['tonnetz_std'] = float(np.std(tonnetz))
except Exception:
    features['tonnetz_mean'] = defaults['tonnetz_mean']
    features['tonnetz_std'] = defaults['tonnetz_std']
```

### 5.3. HNR (Harmonic-to-Noise Ratio) - Dòng 464-465
```python
# HNR - Giữ default (GIỐNG HỆT training)
features['hnr'] = defaults['hnr']
```

---

## 6. LOUDNESS (Độ lớn) - Dòng 456-462

**Đặc trưng**: `loudness`, `loudness_peak` (tính từ RMS)

**Code**:
```python
# Loudness - Tính từ rms_mean (GIỐNG HỆT training)
try:
    features['loudness'] = 20 * np.log10(features.get('rms_mean', 1e-6) + 1e-6)
    features['loudness_peak'] = 20 * np.log10(features.get('rms_max', 1e-6) + 1e-6)
except Exception:
    features['loudness'] = defaults['loudness']
    features['loudness_peak'] = defaults['loudness_peak']
```

**Giải thích**:
- Dòng 458: Tính loudness từ `rms_mean` bằng công thức: `20 * log10(RMS + epsilon)`
- Dòng 459: Tính loudness peak từ `rms_max`
- Đơn vị: decibel (dB)

---

## Tóm tắt vị trí các đặc trưng:

| Loại đặc trưng | Dòng code | Số features |
|---------------|-----------|-------------|
| **Pitch** | 425-454 | 3 (mean, std, range) |
| **MFCC** | 322-331 | 26 (13 × 2) |
| **Spectral Centroid** | 295-302 | 2 (mean, std) |
| **Spectral Rolloff** | 304-311 | 2 (mean, std) |
| **Spectral Bandwidth** | 377-383 | 2 (mean, std) |
| **Spectral Flatness** | 385-391 | 2 (mean, std) |
| **Spectral Contrast** | 342-349 | 2 (mean, std) |
| **Spectral Slope** | 393-400 | 2 (mean, std) |
| **Spectral Kurtosis** | 402-415 | 2 (mean, std) |
| **Spectral Skewness** | 402-415 | 2 (mean, std) |
| **Spectral Flux** | 417-423 | 1 |
| **ZCR** | 313-320 | 2 (mean, std) |
| **RMS** | 285-293 | 4 (mean, std, max, min) |
| **Tempo** | 360-365 | 1 |
| **Duration** | 281-283 | 1 |
| **Onset Strength** | 367-374 | 2 (mean, std) |
| **Chroma** | 333-340 | 2 (mean, std) |
| **Tonnetz** | 351-358 | 2 (mean, std) |
| **HNR** | 464-465 | 1 |
| **Loudness** | 456-462 | 2 (loudness, loudness_peak) |

**Tổng cộng**: ~50+ features

---

## File tương tự:

File `train_audio_features.py` cũng có hàm `extract_audio_features()` với logic giống hệt, được sử dụng để tạo dữ liệu training ban đầu.

