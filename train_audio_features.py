#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script train đặc trưng âm thanh thực tế từ folder trainset
"""
import os
import sys
import codecs
import warnings
from pathlib import Path
import json
import csv
import argparse
import time

# colorama optional for colored output
try:
    from colorama import init as _colorama_init, Fore, Style
    _colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except Exception:
    COLORAMA_AVAILABLE = False
    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        CYAN = ""
    class Style:
        RESET_ALL = ""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from scipy.stats import kurtosis, skew

warnings.filterwarnings('ignore')

# Set encoding for Windows stdout/stderr
if sys.platform.startswith('win'):
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except Exception:
        pass

def compute_spectral_slope(y, sr, n_fft=2048, hop_length=512):
    """
    Compute spectral slope per frame by fitting a line to log-magnitude vs frequency.
    Returns array of slopes (one per frame).
    """
    try:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    except Exception:
        # fallback: zero slopes
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

class AudioFeatureTrainer:
    def __init__(self):
        self.trainset_folder = "trainset"
        self.super_metadata_folder = "super_metadata"
        self.metadata_folder = "metadata"
        Path(self.super_metadata_folder).mkdir(exist_ok=True)

    def get_default_features(self):
        """Return default feature dict (zeros)"""
        return {
            'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0,
            'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0, 'spectral_rolloff_std': 0.0,
            'zcr_mean': 0.0, 'zcr_std': 0.0,
            'mfcc_1_mean': 0.0, 'mfcc_1_std': 0.0, 'mfcc_2_mean': 0.0, 'mfcc_2_std': 0.0,
            'mfcc_3_mean': 0.0, 'mfcc_3_std': 0.0, 'mfcc_4_mean': 0.0, 'mfcc_4_std': 0.0,
            'mfcc_5_mean': 0.0, 'mfcc_5_std': 0.0, 'mfcc_6_mean': 0.0, 'mfcc_6_std': 0.0,
            'mfcc_7_mean': 0.0, 'mfcc_7_std': 0.0, 'mfcc_8_mean': 0.0, 'mfcc_8_std': 0.0,
            'mfcc_9_mean': 0.0, 'mfcc_9_std': 0.0, 'mfcc_10_mean': 0.0, 'mfcc_10_std': 0.0,
            'mfcc_11_mean': 0.0, 'mfcc_11_std': 0.0, 'mfcc_12_mean': 0.0, 'mfcc_12_std': 0.0,
            'mfcc_13_mean': 0.0, 'mfcc_13_std': 0.0,
            'chroma_mean': 0.0, 'chroma_std': 0.0,
            'spectral_contrast_mean': 0.0, 'spectral_contrast_std': 0.0,
            'tonnetz_mean': 0.0, 'tonnetz_std': 0.0,
            'rms_mean': 0.0, 'rms_std': 0.0, 'rms_max': 0.0, 'rms_min': 0.0,
            'tempo': 0.0, 'duration': 0.0, 'loudness': 0.0, 'loudness_peak': 0.0,
            'spectral_bandwidth_mean': 0.0, 'spectral_bandwidth_std': 0.0,
            'spectral_flatness_mean': 0.0, 'spectral_flatness_std': 0.0,
            'hnr': 0.0, 'spectral_slope_mean': 0.0, 'spectral_slope_std': 0.0,
            'spectral_kurtosis_mean': 0.0, 'spectral_kurtosis_std': 0.0,
            'spectral_skewness_mean': 0.0, 'spectral_skewness_std': 0.0,
            'onset_strength_mean': 0.0, 'onset_strength_std': 0.0,
            'spectral_flux': 0.0
        }

    def extract_audio_features(self, audio_path):
        """Extract audio features safely; return dict of features (use defaults on error)."""
        defaults = self.get_default_features()
        try:
            # load audio (prefer soundfile for exact)
            try:
                y, sr = sf.read(audio_path, always_2d=False)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
            except Exception:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
            if y.size == 0:
                raise ValueError("Empty audio")

            # convert to float
            y = librosa.util.normalize(y.astype(float))

            features = {}

            # duration
            duration = float(len(y) / float(sr))
            features['duration'] = duration

            # RMS
            try:
                rms = librosa.feature.rms(y=y)[0]
                features['rms_mean'] = float(np.mean(rms))
                features['rms_std'] = float(np.std(rms))
                features['rms_max'] = float(np.max(rms))
                features['rms_min'] = float(np.min(rms))
            except Exception:
                features.update({k: defaults[k] for k in ['rms_mean','rms_std','rms_max','rms_min']})

            # spectral centroid
            try:
                sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                features['spectral_centroid_mean'] = float(np.mean(sc))
                features['spectral_centroid_std'] = float(np.std(sc))
            except Exception:
                features['spectral_centroid_mean'] = defaults['spectral_centroid_mean']
                features['spectral_centroid_std'] = defaults['spectral_centroid_std']

            # spectral rolloff
            try:
                roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                features['spectral_rolloff_mean'] = float(np.mean(roll))
                features['spectral_rolloff_std'] = float(np.std(roll))
            except Exception:
                features['spectral_rolloff_mean'] = defaults['spectral_rolloff_mean']
                features['spectral_rolloff_std'] = defaults['spectral_rolloff_std']

            # zcr
            try:
                z = librosa.feature.zero_crossing_rate(y)[0]
                features['zcr_mean'] = float(np.mean(z))
                features['zcr_std'] = float(np.std(z))
            except Exception:
                features['zcr_mean'] = defaults['zcr_mean']
                features['zcr_std'] = defaults['zcr_std']

            # mfccs (13)
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                for i in range(13):
                    features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
                    features[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
            except Exception:
                for i in range(13):
                    features[f'mfcc_{i+1}_mean'] = defaults[f'mfcc_{i+1}_mean']
                    features[f'mfcc_{i+1}_std'] = defaults[f'mfcc_{i+1}_std']

            # chroma
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                features['chroma_mean'] = float(np.mean(chroma))
                features['chroma_std'] = float(np.std(chroma))
            except Exception:
                features['chroma_mean'] = defaults['chroma_mean']
                features['chroma_std'] = defaults['chroma_std']

            # spectral contrast
            try:
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features['spectral_contrast_mean'] = float(np.mean(contrast))
                features['spectral_contrast_std'] = float(np.std(contrast))
            except Exception:
                features['spectral_contrast_mean'] = defaults['spectral_contrast_mean']
                features['spectral_contrast_std'] = defaults['spectral_contrast_std']

            # tonnetz
            try:
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
                features['tonnetz_mean'] = float(np.mean(tonnetz))
                features['tonnetz_std'] = float(np.std(tonnetz))
            except Exception:
                features['tonnetz_mean'] = defaults['tonnetz_mean']
                features['tonnetz_std'] = defaults['tonnetz_std']

            # tempo (beat)
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = float(tempo)
            except Exception:
                features['tempo'] = defaults['tempo']

            # onset strength
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                features['onset_strength_mean'] = float(np.mean(onset_env))
                features['onset_strength_std'] = float(np.std(onset_env))
            except Exception:
                features['onset_strength_mean'] = defaults['onset_strength_mean']
                features['onset_strength_std'] = defaults['onset_strength_std']

            # spectral bandwidth & flatness
            try:
                bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                features['spectral_bandwidth_mean'] = float(np.mean(bw))
                features['spectral_bandwidth_std'] = float(np.std(bw))
            except Exception:
                features['spectral_bandwidth_mean'] = defaults['spectral_bandwidth_mean']
                features['spectral_bandwidth_std'] = defaults['spectral_bandwidth_std']

            try:
                flat = librosa.feature.spectral_flatness(y=y)[0]
                features['spectral_flatness_mean'] = float(np.mean(flat))
                features['spectral_flatness_std'] = float(np.std(flat))
            except Exception:
                features['spectral_flatness_mean'] = defaults['spectral_flatness_mean']
                features['spectral_flatness_std'] = defaults['spectral_flatness_std']

            # spectral slope
            try:
                slopes = compute_spectral_slope(y, sr)
                features['spectral_slope_mean'] = float(np.mean(slopes))
                features['spectral_slope_std'] = float(np.std(slopes))
            except Exception:
                features['spectral_slope_mean'] = defaults['spectral_slope_mean']
                features['spectral_slope_std'] = defaults['spectral_slope_std']

            # spectral kurtosis & skewness computed from magnitude STFT across frames
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

            # spectral flux (frame-to-frame change)
            try:
                S = np.abs(librosa.stft(y))
                flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
                features['spectral_flux'] = float(np.mean(flux)) if flux.size > 0 else 0.0
            except Exception:
                features['spectral_flux'] = defaults['spectral_flux']

            # pitch: try pyin, fallback to yin, fallback to zeros
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

            # loudness (simple proxy: dB of RMS)
            try:
                features['loudness'] = 20 * np.log10(features.get('rms_mean', 1e-6) + 1e-6)
                features['loudness_peak'] = 20 * np.log10(features.get('rms_max', 1e-6) + 1e-6)
            except Exception:
                features['loudness'] = defaults['loudness']
                features['loudness_peak'] = defaults['loudness_peak']

            # HNR placeholder (requires more advanced processing) -> keep default
            features['hnr'] = defaults['hnr']

            # merge defaults for any missing keys
            for k, v in defaults.items():
                if k not in features:
                    features[k] = v

            return features

        except Exception as e:
            print(f"⚠️  Lỗi khi trích xuất đặc trưng từ {audio_path}: {e}")
            # return defaults
            return defaults

    def train_trainset_features(self):
        """Train đặc trưng cho trainset (ghép từ clean_testset + noisy_testset)"""
        print("=== Train đặc trưng âm thanh cho Trainset (clean + noisy) ===\n")

        clean_path = os.path.join(self.metadata_folder, "clean_testset.csv")
        noisy_path = os.path.join(self.metadata_folder, "noisy_testset.csv")

        found = []
        if os.path.exists(clean_path):
            df_clean = pd.read_csv(clean_path, encoding='utf-8')
            df_clean = df_clean.assign(split='clean')
            found.append(df_clean)
            print(f"Đã đọc {len(df_clean)} records từ clean_testset.csv")
        else:
            print(f"  ⚠️  Không tìm thấy {clean_path}")

        if os.path.exists(noisy_path):
            df_noisy = pd.read_csv(noisy_path, encoding='utf-8')
            df_noisy = df_noisy.assign(split='noisy')
            found.append(df_noisy)
            print(f"Đã đọc {len(df_noisy)} records từ noisy_testset.csv")
        else:
            print(f"  ⚠️  Không tìm thấy {noisy_path}")

        if not found:
            raise FileNotFoundError(f"Missing metadata files: {clean_path} and {noisy_path}")

        # concatenate both dataframes
        df = pd.concat(found, ignore_index=True)
        print(f"Tổng cộng {len(df)} records (clean + noisy)")

        super_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Train đặc trưng trainset"):
            audio_name = row.get('audio_name') or row.get('filename') or row.get('file') or None
            if not audio_name:
                print(f"  ⚠️  Bỏ qua record {idx}: không tìm tên file")
                continue
            audio_path = os.path.join(self.trainset_folder, audio_name)
            if not os.path.exists(audio_path):
                print(f"  ⚠️  Không tìm thấy file audio: {audio_path} -> dùng giá trị mặc định")
                features = self.get_default_features()
            else:
                features = self.extract_audio_features(audio_path)
            # merge metadata row (as dict) with features
            merged = {}
            # keep original metadata columns
            for c in df.columns:
                merged[c] = row[c]
            # add features
            merged.update(features)
            super_data.append(merged)

        super_df = pd.DataFrame(super_data)
        output_file = os.path.join(self.super_metadata_folder, "trainset.csv")
        super_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✅ Đã tạo super metadata: {output_file}")
        print(f"📊 Số cột: {len(super_df.columns)}")
        print(f"📊 Số dòng: {len(super_df)}")
        return super_df

    def update_other_datasets(self):
        """Cập nhật clean_testset và noisy_testset với giá trị mặc định"""
        print("\n=== Cập nhật Clean Testset và Noisy Testset ===")
        default_feats = self.get_default_features()
        for in_path, out_name in [
            (os.path.join(self.metadata_folder, "clean_testset.csv"), "clean_testset.csv"),
            (os.path.join(self.metadata_folder, "noisy_testset.csv"), "noisy_testset.csv")
        ]:
            if not os.path.exists(in_path):
                print(f"  ⚠️  Không tìm thấy {in_path}, bỏ qua")
                continue
            df = pd.read_csv(in_path, encoding='utf-8')
            rows = []
            for _, row in df.iterrows():
                merged = {}
                for c in df.columns:
                    merged[c] = row[c]
                merged.update(default_feats)
                rows.append(merged)
            out_df = pd.DataFrame(rows)
            out_path = os.path.join(self.super_metadata_folder, out_name)
            out_df.to_csv(out_path, index=False, encoding='utf-8')
            print(f"  -> Đã tạo {out_path} ({len(out_df)} rows)")

    def run_training(self):
        """Full pipeline"""
        super_df = self.train_trainset_features()
        self.update_other_datasets()
        return super_df

def cprint(msg, color=None):
    if not COLORAMA_AVAILABLE or color is None:
        print(msg)
    else:
        print(f"{color}{msg}{Style.RESET_ALL}")

# Thêm helper chuyển đối tượng sang định dạng serializable
def to_serializable(obj):
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, (list, dict, str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)

# Thêm helper tìm file audio trong thư mục trainset
def find_audio_file(root: Path, name: str):
    if not name:
        return None
    p = root / name
    if p.exists():
        return p
    for ext in (".mp3", ".wav", ".flac", ".ogg"):
        p2 = root / (name + ext)
        if p2.exists():
            return p2
    lowered = name.lower()
    for f in root.iterdir():
        if f.name.lower() == lowered:
            return f
    for f in root.iterdir():
        if f.name.lower().startswith(lowered):
            return f
    return None

# Thay thế hàm main bằng phiên bản hỗ trợ chạy theo batch (mỗi step tối đa batch_size file)
def main():
    parser = argparse.ArgumentParser(description="Extract features in batches (steps).")
    parser.add_argument("--batch-size", type=int, default=150, help="Số file tối đa mỗi step (mặc định 150)")
    parser.add_argument("--step", type=int, default=None, help="Chạy step cụ thể (1-based). Nếu không set, dùng --start-index")
    parser.add_argument("--start-index", type=int, default=None, help="Chỉ xử lý từ index bắt đầu (0-based).")
    parser.add_argument("--max-files", type=int, default=None, help="Giới hạn tổng số file xử lý từ start (tùy chọn).")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ các batch tuần tự (ghi nhiều file part).")
    parser.add_argument("--append-master", action="store_true", help="Ghi nối kết quả vào super_metadata/ket_qua_cuoi.csv master.")
    parser.add_argument("--from-step", type=int, default=None, help="Khi --all set, bắt đầu từ step này (1-based).")
    args = parser.parse_args()

    trainer = AudioFeatureTrainer()

    trainset_dir = Path(trainer.trainset_folder)
    if not trainset_dir.exists():
        cprint(f"Không tìm thấy thư mục trainset: {trainset_dir}", Fore.RED)
        return

    csv_in = Path(trainer.metadata_folder) / 'trainset.csv'
    if not csv_in.exists():
        cprint(f"Không tìm thấy metadata: {csv_in}", Fore.RED)
        return

    # đọc toàn bộ hàng metadata (nhẹ, dùng slicing cho batch)
    with csv_in.open(newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    total = len(reader)
    if total == 0:
        cprint("Không có hàng trong trainset.csv", Fore.YELLOW)
        return

    batch_size = max(1, args.batch_size)

    # xác định các batch để chạy
    batches = []
    if args.all:
        # tạo list các (start, end, step_no)
        step_no = 1
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batches.append((start, end, step_no))
            step_no += 1
        # apply from-step filter
        if args.from_step:
            batches = [b for b in batches if b[2] >= max(1, args.from_step)]
    else:
        if args.step is not None:
            step = max(1, args.step)
            start = (step - 1) * batch_size
            end = min(start + batch_size, total)
            batches.append((start, end, step))
        else:
            if args.start_index is None:
                start = 0
            else:
                start = max(0, args.start_index)
            end = min(start + batch_size, total)
            batches.append((start, end, 1 + start // batch_size))

    # prepare master accumulators
    master_rows = []
    master_fieldnames = set()
    metadata_fields = list(reader[0].keys()) if total>0 else ['audio_name']

    # cache defaults to avoid repeated calls
    DEFAULT_FEATURES = trainer.get_default_features()

    # process each batch
    for start, end, step_no in batches:
        cprint(f"\n--- Batch {step_no}: items {start}..{end-1} (count {end-start}) ---", Fore.CYAN)
        out_rows = []
        processed = 0
        failed = 0
        t0 = time.time()

        # use tqdm for per-batch progress
        for i in tqdm(range(start, end), desc=f"Batch {step_no}", unit="file"):
            row = reader[i]
            audio_name = row.get('audio_name') or row.get('audio') or row.get('filename')
            if not audio_name:
                cprint(f"[{i}] Bỏ qua: không có audio_name", Fore.YELLOW)
                failed += 1
                continue

            audio_path = find_audio_file(trainset_dir, audio_name)
            if not audio_path:
                cprint(f"[{i}] Không tìm thấy file: {audio_name}", Fore.YELLOW)
                out_entry = dict(row)
                # fill default features columns (no features_json)
                for k, v in DEFAULT_FEATURES.items():
                    out_entry[k] = v
                out_rows.append(out_entry)
                master_rows.append(out_entry)
                master_fieldnames.update(out_entry.keys())
                failed += 1
                continue

            try:
                feats = trainer.extract_audio_features(str(audio_path))
                # normalize feature values and explode into columns (no features_json)
                feats_serial = {k: to_serializable(v) for k, v in feats.items()}
                out_entry = dict(row)
                out_entry.update(feats_serial)        # explode features into columns
                out_rows.append(out_entry)
                master_rows.append(out_entry)
                master_fieldnames.update(out_entry.keys())
                processed += 1
            except Exception as e:
                cprint(f"[{i}] Lỗi extract {audio_name}: {e}", Fore.RED)
                out_entry = dict(row)
                for k, v in DEFAULT_FEATURES.items():
                    out_entry[k] = v
                out_rows.append(out_entry)
                master_rows.append(out_entry)
                master_fieldnames.update(out_entry.keys())
                failed += 1

        dt = time.time() - t0
        cprint(f"Batch {step_no} done. processed={processed} failed={failed} time={dt:.1f}s", Fore.GREEN)

        # ensure we don't include the JSON column
        if 'features_json' in master_fieldnames:
            master_fieldnames.discard('features_json')

        # determine full field list: metadata fields + feature fields (excluding duplicates)
        feature_fields = sorted([f for f in master_fieldnames if f not in metadata_fields])
        fieldnames_part = metadata_fields + feature_fields

        # write part CSV (use quoting to handle commas inside values)
        part_csv = Path(trainer.super_metadata_folder) / f'ket_qua_cuoi_part_{step_no:03d}.csv'
        with part_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_part, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for r in out_rows:
                row_for_write = {k: (r.get(k, '') if r.get(k, '') is not None else '') for k in fieldnames_part}
                writer.writerow(row_for_write)
        cprint(f"Batch {step_no} written -> {part_csv}", Fore.CYAN)

        # if not running all, break after first requested batch
        if not args.all:
            break

    # write/append master CSV if requested
    if args.append_master or args.all:
        # ensure we don't include the JSON column
        if 'features_json' in master_fieldnames:
            master_fieldnames.discard('features_json')

        master_csv = Path(trainer.super_metadata_folder) / 'ket_qua_cuoi.csv'
        feature_fields = sorted([f for f in master_fieldnames if f not in metadata_fields])
        fieldnames_master = metadata_fields + feature_fields
        mode = 'a' if master_csv.exists() and args.append_master else 'w'
        write_header = not (master_csv.exists() and args.append_master)
        with master_csv.open(mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_master, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
            if write_header:
                writer.writeheader()
            for r in master_rows:
                row_for_write = {k: (r.get(k, '') if r.get(k, '') is not None else '') for k in fieldnames_master}
                writer.writerow(row_for_write)
        cprint(f"Master results updated -> {master_csv}", Fore.GREEN)

    cprint("All done.", Fore.CYAN)

if __name__ == "__main__":
    main()
