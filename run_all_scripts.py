#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tổng hợp để chạy tất cả các chức năng
"""

import os
import subprocess
import sys

def run_script(script_name, description):
    """Chạy một script Python"""
    print(f"\n{'='*60}")
    print(f"Chạy: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("✅ Thành công!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ Lỗi!")
            if result.stderr:
                print("Error:")
                print(result.stderr)
                
    except Exception as e:
        print(f"❌ Lỗi khi chạy script: {e}")

def main():
    """Hàm chính - chạy tất cả scripts"""
    print("🚀 Bắt đầu chạy tất cả scripts...")
    
    # Danh sách các script cần chạy theo thứ tự
    scripts = [
        ("create_speaker_database.py", "Tạo database speaker với tên đàn ông Việt Nam"),
        ("test_super_metadata.py", "Tạo super metadata đơn giản với các trường mặc định"),
    ]
    
    # Chạy từng script
    for script_name, description in scripts:
        if os.path.exists(script_name):
            run_script(script_name, description)
        else:
            print(f"⚠️  Không tìm thấy script: {script_name}")
    
    print(f"\n{'='*60}")
    print("🎉 Hoàn thành tất cả scripts!")
    print("📁 Các folder đã tạo:")
    print("  - speaker_database.csv (database speaker)")
    print("  - super_metadata/ (metadata mở rộng)")
    print("  - data_check/ (folder upload audio)")
    print("  - backup_deleted_files/ (backup files đã xóa)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
