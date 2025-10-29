#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script khởi động Algorithm Testing App
"""

import os
import sys

def check_requirements():
    """Kiểm tra các yêu cầu cần thiết"""
    print("Kiem tra yeu cau he thong...")
    
    # Kiểm tra Python version
    if sys.version_info < (3, 7):
        print("ERROR - Can Python 3.7 tro len!")
        return False
    
    # Kiểm tra các file cần thiết
    required_files = [
        "super_metadata/male_only_merged.csv",
        "speaker_database.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR - Thieu cac file sau:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nHuong dan:")
        print("1. Chay: python merge_male_only.py")
        print("2. Hoac: python run_training.py")
        return False
    
    print("OK - Tat ca yeu cau da dap ung!")
    return True

def install_requirements():
    """Cài đặt requirements"""
    print("Cai dat thu vien can thiet...")
    os.system("pip install -r requirements.txt")

def start_algorithm_testing_app():
    """Khởi động algorithm testing app"""
    print("Khoi dong Algorithm Testing App...")
    print("=" * 60)
    
    try:
        from voice_algorithm_testing_app import main
        main()
        
    except ImportError as e:
        print(f"ERROR - Loi import: {e}")
        print("Cai dat requirements:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"ERROR - Loi khoi dong: {e}")

def main():
    """Hàm chính"""
    print("ViSpeech - Algorithm Testing & Comparison")
    print("=" * 60)
    print("TINH NANG:")
    print("- Test 12 thuat toan ML khac nhau")
    print("- So sanh hieu suat va thoi gian")
    print("- Phan tich diem manh/yeu cua tung thuat toan")
    print("- Bao cao chi tiet ket qua")
    print("=" * 60)
    
    # Kiểm tra yêu cầu
    if not check_requirements():
        return
    
    # Hỏi có muốn cài đặt requirements không
    response = input("\nCo muon cai dat/cap nhat thu vien khong? (y/N): ")
    if response.lower() == 'y':
        install_requirements()
    
    # Khởi động algorithm testing app
    start_algorithm_testing_app()

if __name__ == "__main__":
    main()
