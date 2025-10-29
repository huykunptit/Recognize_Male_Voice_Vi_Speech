#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script khởi động ViSpeech Desktop App Final
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

def start_final_app():
    """Khởi động final desktop app"""
    print("Khoi dong ViSpeech Desktop App Final...")
    print("=" * 60)
    
    try:
        # Hỏi người dùng chọn loại app
        print("Chon loai ung dung:")
        print("1. Desktop App Final (co ban)")
        print("2. Auto Regional Detection App (tu dong phat hien vung mien)")
        
        choice = input("\nNhap lua chon (1-2): ").strip()
        
        if choice == "1":
            from voice_desktop_app_final import main
            main()
        elif choice == "2":
            from voice_auto_regional_app import main
            main()
        else:
            print("Lua chon khong hop le, khoi dong app co ban...")
            from voice_desktop_app_final import main
            main()
        
    except ImportError as e:
        print(f"ERROR - Loi import: {e}")
        print("Cai dat requirements:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"ERROR - Loi khoi dong: {e}")

def main():
    """Hàm chính"""
    print("ViSpeech - Voice Comparison Desktop App (Final)")
    print("=" * 60)
    
    # Kiểm tra yêu cầu
    if not check_requirements():
        return
    
    # Hỏi có muốn cài đặt requirements không
    response = input("\nCo muon cai dat/cap nhat thu vien khong? (y/N): ")
    if response.lower() == 'y':
        install_requirements()
    
    # Khởi động final desktop app
    start_final_app()

if __name__ == "__main__":
    main()
