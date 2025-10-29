#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script khởi động Regional Voice Comparison App
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
        "trainset",
        "metadata/trainset.csv",
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
        return False
    
    print("OK - Tat ca yeu cau da dap ung!")
    return True

def install_requirements():
    """Cài đặt requirements"""
    print("Cai dat thu vien can thiet...")
    os.system("pip install -r requirements.txt")

def start_regional_app():
    """Khởi động regional desktop app"""
    print("Khoi dong Regional Voice Comparison App...")
    print("=" * 60)
    
    try:
        from voice_regional_app import main
        main()
        
    except ImportError as e:
        print(f"ERROR - Loi import: {e}")
        print("Cai dat requirements:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"ERROR - Loi khoi dong: {e}")

def main():
    """Hàm chính"""
    print("ViSpeech - Regional Voice Comparison App")
    print("=" * 60)
    print("TINH NANG MOI:")
    print("- Loc du lieu theo vung mien (North, Central, South)")
    print("- Training rieng cho tung vung mien")
    print("- So sanh giong noi theo vung mien da chon")
    print("- Hien thi ket qua co phan loai vung mien")
    print("=" * 60)
    
    # Kiểm tra yêu cầu
    if not check_requirements():
        return
    
    # Hỏi có muốn cài đặt requirements không
    response = input("\nCo muon cai dat/cap nhat thu vien khong? (y/N): ")
    if response.lower() == 'y':
        install_requirements()
    
    # Khởi động regional app
    start_regional_app()

if __name__ == "__main__":
    main()
