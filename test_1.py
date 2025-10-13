#!/usr/bin/env python3
"""Quick feature extraction smoke test for one audio file."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_audio_features import AudioFeatureTrainer

def main():
    trainer = AudioFeatureTrainer()
    # pick a small, early file from trainset
    test_audio = ROOT / 'trainset' / 'ViSpeech_01951.mp3'
    if not test_audio.exists():
        print(f"Test audio not found: {test_audio}")
        return

    print(f"Running feature extraction on: {test_audio}")
    feats = trainer.extract_audio_features(str(test_audio))

    # print a compact view
    keys = sorted(feats.keys())
    for k in keys:
        print(f"{k}: {feats[k]}")

if __name__ == '__main__':
    main()
