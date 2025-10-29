#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script khởi động Auto Regional Voice Detection App
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
        "speaker_database.csv",
        "trainset",
        "metadata/trainset.csv"
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

def start_auto_regional_app():
    """Khởi động auto regional desktop app"""
    print("Khoi dong Auto Regional Voice Detection App...")
    print("=" * 60)
    
    try:
        from voice_auto_regional_app import main
        main()
        
    except ImportError as e:
        print(f"ERROR - Loi import: {e}")
        print("Cai dat requirements:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"ERROR - Loi khoi dong: {e}")

def main():
    """Hàm chính"""
    print("ViSpeech - Auto Regional Voice Detection App")
    print("=" * 60)
    print("TINH NANG MOI:")
    print("- Tu dong phat hien vung mien (North, Central, South)")
    print("- Loc ket qua so sanh theo vung mien da phat hien")
    print("- Hien thi do tin cay va xac suat cac vung mien")
    print("- Luu ket qua phat hien vao JSON")
    print("- So sanh chinh xac hon voi vung mien tuong ung")
    print("=" * 60)
    
    # Kiểm tra yêu cầu
    if not check_requirements():
        return
    
    # Hỏi có muốn cài đặt requirements không
    response = input("\nCo muon cai dat/cap nhat thu vien khong? (y/N): ")
    if response.lower() == 'y':
        install_requirements()
    
    # Khởi động auto regional app
    start_auto_regional_app()

if __name__ == "__main__":
    main()
