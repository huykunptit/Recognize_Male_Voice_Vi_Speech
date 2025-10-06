import os
import pandas as pd
import shutil
from pathlib import Path

def load_metadata(csv_path):
    """Đọc file metadata CSV"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Đã đọc {len(df)} records từ {csv_path}")
        return df
    except Exception as e:
        print(f"Lỗi khi đọc file {csv_path}: {e}")
        return None

def get_male_files(df):
    """Lấy danh sách các file audio có gender = 'Male'"""
    if df is None:
        return []
    
    male_files = df[df['gender'] == 'Male']['audio_name'].tolist()
    print(f"Tìm thấy {len(male_files)} file audio có gender = 'Male'")
    return male_files

def filter_audio_files(folder_path, male_files, backup_folder=None):
    """
    Lọc các file audio trong folder, chỉ giữ lại các file có trong male_files
    Xóa các file không hợp lệ
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} không tồn tại!")
        return
    
    # Tạo backup folder nếu được chỉ định
    if backup_folder:
        backup_path = Path(backup_folder)
        backup_path.mkdir(exist_ok=True)
        print(f"Backup folder: {backup_path}")
    
    # Lấy danh sách tất cả file audio trong folder
    audio_files = list(folder.glob("*.mp3"))
    print(f"Tìm thấy {len(audio_files)} file audio trong {folder_path}")
    
    # Đếm số file cần xóa và số file cần giữ
    files_to_keep = []
    files_to_delete = []
    
    for audio_file in audio_files:
        if audio_file.name in male_files:
            files_to_keep.append(audio_file)
        else:
            files_to_delete.append(audio_file)
    
    print(f"Sẽ giữ lại: {len(files_to_keep)} file")
    print(f"Sẽ xóa: {len(files_to_delete)} file")
    
    # Xác nhận trước khi xóa
    if files_to_delete:
        response = input(f"Bạn có chắc chắn muốn xóa {len(files_to_delete)} file không hợp lệ? (y/N): ")
        if response.lower() != 'y':
            print("Hủy bỏ thao tác xóa file.")
            return
        
        # Backup các file trước khi xóa (nếu có backup folder)
        if backup_folder:
            print("Đang backup các file sẽ bị xóa...")
            for file_to_delete in files_to_delete:
                backup_file = backup_path / file_to_delete.name
                shutil.copy2(file_to_delete, backup_file)
            print(f"Đã backup {len(files_to_delete)} file vào {backup_path}")
        
        # Xóa các file không hợp lệ
        deleted_count = 0
        for file_to_delete in files_to_delete:
            try:
                file_to_delete.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Lỗi khi xóa file {file_to_delete.name}: {e}")
        
        print(f"Đã xóa {deleted_count} file không hợp lệ")
    else:
        print("Không có file nào cần xóa!")

def main():
    """Hàm chính"""
    print("=== Script lọc file audio theo giới tính Male ===\n")
    
    # Đường dẫn các file và folder
    clean_metadata_path = "metadata/clean_testset.csv"
    noisy_metadata_path = "metadata/noisy_testset.csv"
    clean_folder = "clean_testset"
    noisy_folder = "noisy_testset"
    backup_folder = "backup_deleted_files"
    
    # Kiểm tra các file metadata có tồn tại không
    if not os.path.exists(clean_metadata_path):
        print(f"Không tìm thấy file {clean_metadata_path}")
        return
    
    if not os.path.exists(noisy_metadata_path):
        print(f"Không tìm thấy file {noisy_metadata_path}")
        return
    
    # Đọc metadata
    print("1. Đọc metadata...")
    clean_df = load_metadata(clean_metadata_path)
    noisy_df = load_metadata(noisy_metadata_path)
    
    if clean_df is None or noisy_df is None:
        print("Không thể đọc metadata!")
        return
    
    # Lấy danh sách file Male
    print("\n2. Lọc danh sách file Male...")
    clean_male_files = get_male_files(clean_df)
    noisy_male_files = get_male_files(noisy_df)
    
    # Thống kê
    print(f"\nThống kê:")
    print(f"Clean testset - Tổng: {len(clean_df)}, Male: {len(clean_male_files)}")
    print(f"Noisy testset - Tổng: {len(noisy_df)}, Male: {len(noisy_male_files)}")
    
    # Lọc file trong clean_testset
    print(f"\n3. Lọc file trong {clean_folder}...")
    filter_audio_files(clean_folder, clean_male_files, backup_folder)
    
    # Lọc file trong noisy_testset
    print(f"\n4. Lọc file trong {noisy_folder}...")
    filter_audio_files(noisy_folder, noisy_male_files, backup_folder)
    
    print("\n=== Hoàn thành! ===")
    print("Các file audio không hợp lệ đã được xóa.")
    print("Các file đã xóa được backup trong folder 'backup_deleted_files' (nếu có).")

if __name__ == "__main__":
    main()
