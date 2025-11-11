#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tạo biểu đồ kiến trúc hệ thống ViSpeech
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
    
    # Kiểm tra matplotlib
    try:
        import matplotlib
        print("✓ matplotlib da cai dat")
    except ImportError:
        print("❌ matplotlib chua cai dat")
        return False
    
    print("OK - Tat ca yeu cau da dap ung!")
    return True

def install_requirements():
    """Cài đặt requirements"""
    print("Cai dat thu vien can thiet...")
    os.system("pip install matplotlib")

def create_diagrams():
    """Tạo biểu đồ kiến trúc"""
    print("Tao bieu do kien truc he thong...")
    print("=" * 60)
    
    try:
        from create_architecture_diagrams import main
        main()
        
    except ImportError as e:
        print(f"ERROR - Loi import: {e}")
        print("Cai dat requirements:")
        print("  pip install matplotlib")
    except Exception as e:
        print(f"ERROR - Loi tao bieu do: {e}")

def main():
    """Hàm chính"""
    print("ViSpeech - Architecture Diagram Generator")
    print("=" * 60)
    print("TINH NANG:")
    print("- Tao bieu do kien truc tong quan")
    print("- Tao bieu do kien truc mo hinh")
    print("- Tao bieu do luong du lieu")
    print("- Hien thi moi quan he giua cac thanh phan")
    print("=" * 60)
    
    # Kiểm tra yêu cầu
    if not check_requirements():
        response = input("\nCo muon cai dat matplotlib khong? (y/N): ")
        if response.lower() == 'y':
            install_requirements()
        else:
            return
    
    # Tạo biểu đồ
    create_diagrams()

if __name__ == "__main__":
    main()
