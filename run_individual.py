#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy từng script riêng lẻ để tránh lỗi encoding
"""

import os
import sys

def run_script_safely(script_name):
    """Chạy script một cách an toàn"""
    print(f"Chạy script: {script_name}")
    print("=" * 50)
    
    try:
        # Import và chạy script
        if script_name == "create_speaker_database.py":
            from create_speaker_database import create_speaker_database
            create_speaker_database()
            
        elif script_name == "test_super_metadata.py":
            from test_super_metadata import create_simple_super_metadata
            create_simple_super_metadata()
            
        else:
            print(f"Script {script_name} không được hỗ trợ")
            return False
            
        print(f"✅ Hoàn thành {script_name}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi chạy {script_name}: {e}")
        return False

def main():
    """Hàm chính"""
    print("🚀 Chạy từng script riêng lẻ...\n")
    
    # Danh sách script cần chạy
    scripts = [
        "create_speaker_database.py",
        "test_super_metadata.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            success = run_script_safely(script)
            if success:
                print(f"✅ {script} - Thành công")
            else:
                print(f"❌ {script} - Thất bại")
        else:
            print(f"⚠️  Không tìm thấy: {script}")
        
        print("\n" + "="*60 + "\n")
    
    print("🎉 Hoàn thành tất cả scripts!")

if __name__ == "__main__":
    main()
