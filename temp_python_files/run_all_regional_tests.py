#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chạy tất cả các test regional classifiers và tạo báo cáo tổng hợp
"""

import subprocess
import sys
import os
import pandas as pd
from datetime import datetime

def run_test(script_name, description):
    """Chạy một test script"""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 phút timeout
        )
        
        if result.returncode == 0:
            print(f"✓ {description} - THANH CONG")
            return True, result.stdout
        else:
            print(f"✗ {description} - LOI")
            print(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"✗ {description} - TIMEOUT (>10 phut)")
        return False, "Timeout"
    except Exception as e:
        print(f"✗ {description} - EXCEPTION: {e}")
        return False, str(e)

def create_summary_report():
    """Tạo báo cáo tổng hợp từ các file CSV kết quả"""
    print("\n" + "=" * 80)
    print("CREATING SUMMARY REPORT")
    print("=" * 80)
    
    # Tìm tất cả file CSV kết quả
    csv_files = []
    if os.path.exists("regional_classifier_comparison.csv"):
        csv_files.append(("All Algorithms", "regional_classifier_comparison.csv"))
    
    # Có thể thêm các file CSV từ các test riêng lẻ nếu có
    
    if not csv_files:
        print("Khong tim thay file CSV ket qua!")
        return
    
    # Tạo báo cáo
    report = []
    report.append("=" * 80)
    report.append("REGIONAL CLASSIFIER TESTING SUMMARY")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    for name, csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            report.append(f"\n{name}:")
            report.append("-" * 80)
            report.append(df.to_string(index=False))
            report.append("")
            
            # Top 3
            if 'Test_Accuracy' in df.columns:
                top3 = df.nlargest(3, 'Test_Accuracy')
                report.append(f"\nTop 3 algorithms:")
                report.append(top3.to_string(index=False))
        except Exception as e:
            report.append(f"Error reading {csv_file}: {e}")
    
    # Lưu báo cáo
    report_text = "\n".join(report)
    report_file = f"regional_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nDa luu bao cao vao: {report_file}")
    print("\n" + report_text)

def main():
    """Hàm chính"""
    print("=" * 80)
    print("REGIONAL CLASSIFIER TESTING - ALL TESTS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Danh sách các test
    tests = [
        ("test_regional_classifiers.py", "Test All Algorithms"),
        # Các test riêng lẻ (optional - comment out nếu không muốn chạy)
        # ("test_xgboost_regional.py", "Test XGBoost"),
        # ("test_lightgbm_regional.py", "Test LightGBM"),
        # ("test_svm_regional.py", "Test SVM"),
        # ("test_neural_network_regional.py", "Test Neural Network"),
        # ("test_ensemble_regional.py", "Test Ensemble"),
        # ("test_knn_regional.py", "Test K-NN"),
        # ("test_random_forest_variants.py", "Test Random Forest Variants"),
    ]
    
    results = []
    
    for script, description in tests:
        success, output = run_test(script, description)
        results.append({
            'test': description,
            'script': script,
            'success': success,
            'output': output[:500] if output else ""  # Giới hạn output
        })
    
    # Tạo summary
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"\nTotal tests: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    print("\nDetails:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['test']}")
    
    # Tạo báo cáo tổng hợp
    create_summary_report()
    
    print("\n" + "=" * 80)
    print(f"FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()

