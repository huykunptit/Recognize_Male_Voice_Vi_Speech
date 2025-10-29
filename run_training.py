
import os
import sys

REQUIRED_PKGS = ["librosa", "numpy", "pandas", "soundfile", "tqdm", "scipy"]

def check_dependencies():
    missing = []
    for pkg in REQUIRED_PKGS:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    if missing:
        print("❌ Thiếu thư viện Python:", ", ".join(missing))
        print("👉 Cài nhanh (PowerShell hoặc cmd):")
        print("   pip install " + " ".join(missing))
        print("Hoặc dùng file requirements.txt:")
        print("   pip install -r requirements.txt")
        return False
    return True

def run_training_safely():
    """Chạy training một cách an toàn"""
    print("🚀 Bắt đầu train đặc trưng âm thanh...")
    print("=" * 60)
    
    try:
        from train_audio_features import AudioFeatureTrainer
        
        trainer = AudioFeatureTrainer()
        trainer.run_training()
        
        print("\n✅ Training hoàn thành thành công!")
        return True
        
    except Exception as e:
        print(f"\n❌ Lỗi khi training: {e}")
        print("💡 Hãy kiểm tra:")
        print("  - Folder 'trainset' có tồn tại không")
        print("  - File 'metadata/trainset.csv' có tồn tại không")
        print("  - Đã cài đặt đầy đủ thư viện: pip install -r requirements.txt")
        return False

def main():
    """Hàm chính"""
    if not check_dependencies():
        return

    print("ViSpeech - Training Dac trung Am thanh")
    print("=" * 60)
    
    # Kiểm tra các file cần thiết
    required_files = [
        "trainset",
        "metadata/trainset.csv",
        "metadata/clean_testset.csv", 
        "metadata/noisy_testset.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Thiếu các file/folder sau:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n💡 Hãy đảm bảo có đầy đủ dữ liệu trước khi chạy training!")
        return
    
    print("✅ Tất cả file cần thiết đã có")
    print("\n🎯 Bắt đầu training...")
    
    # Chạy training
    success = run_training_safely()
    
    if success:
        print("\n🎉 Training hoàn thành!")
        print("📁 Kết quả được lưu trong folder 'super_metadata/'")
        print("📋 Các file đã tạo:")
        if os.path.exists("super_metadata"):
            for file in os.listdir("super_metadata"):
                if file.endswith('.csv'):
                    print(f"  - {file}")
    else:
        print("\n❌ Training thất bại!")
        print("💡 Hãy kiểm tra lại và thử lại")

if __name__ == "__main__":
    main()
