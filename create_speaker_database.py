import pandas as pd
import os
from pathlib import Path

def get_unique_speakers():
    """Lấy danh sách speaker duy nhất từ tất cả file metadata"""
    speakers = set()
    
    # Đọc từ clean_testset
    if os.path.exists("metadata/clean_testset.csv"):
        df = pd.read_csv("metadata/clean_testset.csv")
        speakers.update(df['speaker'].unique())
    
    # Đọc từ noisy_testset  
    if os.path.exists("metadata/noisy_testset.csv"):
        df = pd.read_csv("metadata/noisy_testset.csv")
        speakers.update(df['speaker'].unique())
    
    # Đọc từ trainset
    if os.path.exists("metadata/trainset.csv"):
        df = pd.read_csv("metadata/trainset.csv")
        speakers.update(df['speaker'].unique())
    
    return sorted(list(speakers))

def generate_vietnamese_male_names():
    """Tạo danh sách tên đàn ông Việt Nam"""
    vietnamese_male_names = [
        "Nguyễn Văn An", "Trần Văn Bình", "Lê Văn Cường", "Phạm Văn Dũng", "Hoàng Văn Em",
        "Vũ Văn Phong", "Đặng Văn Giang", "Bùi Văn Hải", "Phan Văn Hùng", "Võ Văn Khánh",
        "Ngô Văn Lâm", "Đinh Văn Minh", "Lý Văn Nam", "Đỗ Văn Oanh", "Tôn Văn Phúc",
        "Hồ Văn Quang", "Lưu Văn Sơn", "Đào Văn Tân", "Vương Văn Uy", "Chu Văn Vinh",
        "Lâm Văn Xuân", "Nguyễn Đức An", "Trần Minh Bảo", "Lê Hữu Cảnh", "Phạm Quang Dân",
        "Hoàng Văn Đức", "Vũ Minh Giang", "Đặng Văn Hòa", "Bùi Đức Khánh", "Phan Văn Lâm",
        "Võ Minh Nam", "Ngô Đức Phúc", "Đinh Văn Quang", "Lý Minh Sơn", "Đỗ Văn Tân",
        "Tôn Đức Uy", "Hồ Minh Vinh", "Lưu Văn Xuân", "Đào Đức An", "Vương Minh Bảo",
        "Chu Văn Cảnh", "Lâm Đức Dân", "Nguyễn Minh Đức", "Trần Văn Giang", "Lê Đức Hòa",
        "Phạm Minh Khánh", "Hoàng Văn Lâm", "Vũ Đức Nam", "Đặng Minh Phúc", "Bùi Văn Quang",
        "Phan Đức Sơn", "Võ Minh Tân", "Ngô Văn Uy", "Đinh Đức Vinh", "Lý Minh Xuân",
        "Đỗ Văn An", "Tôn Minh Bảo", "Hồ Đức Cảnh", "Lưu Minh Dân", "Đào Văn Đức",
        "Vương Đức Giang", "Chu Minh Hòa", "Lâm Văn Khánh", "Nguyễn Đức Lâm", "Trần Minh Nam",
        "Lê Văn Phúc", "Phạm Đức Quang", "Hoàng Minh Sơn", "Vũ Văn Tân", "Đặng Đức Uy",
        "Bùi Minh Vinh", "Phan Văn Xuân", "Võ Đức An", "Ngô Minh Bảo", "Đinh Văn Cảnh",
        "Lý Đức Dân", "Đỗ Minh Đức", "Tôn Văn Giang", "Hồ Đức Hòa", "Lưu Minh Khánh",
        "Đào Văn Lâm", "Vương Đức Nam", "Chu Minh Phúc", "Lâm Văn Quang", "Nguyễn Đức Sơn",
        "Trần Minh Tân", "Lê Văn Uy", "Phạm Đức Vinh", "Hoàng Minh Xuân", "Vũ Văn An",
        "Đặng Minh Bảo", "Bùi Đức Cảnh", "Phan Minh Dân", "Võ Văn Đức", "Ngô Đức Giang",
        "Đinh Minh Hòa", "Lý Văn Khánh", "Đỗ Đức Lâm", "Tôn Minh Nam", "Hồ Văn Phúc",
        "Lưu Đức Quang", "Đào Minh Sơn", "Vương Văn Tân", "Chu Đức Uy", "Lâm Minh Vinh"
    ]
    return vietnamese_male_names

def create_speaker_database():
    """Tạo database speaker với tên đàn ông Việt Nam"""
    print("=== Tạo Database Speaker ===\n")
    
    # Lấy danh sách speaker duy nhất
    speakers = get_unique_speakers()
    print(f"Tìm thấy {len(speakers)} speaker duy nhất")
    
    # Tạo danh sách tên đàn ông Việt Nam
    vietnamese_names = generate_vietnamese_male_names()
    print(f"Tạo {len(vietnamese_names)} tên đàn ông Việt Nam")
    
    # Tạo mapping
    speaker_data = []
    for i, speaker in enumerate(speakers):
        # Lấy tên theo index, nếu hết tên thì lặp lại
        name = vietnamese_names[i % len(vietnamese_names)]
        dialect = speaker
        speaker_data.append({
            'speaker_id': speaker,
            'vietnamese_name': name,
            'gender': 'Male',
            'dialect': dialect,
            'region': 'Vietnam',
            'language': 'Vietnamese'
        })
    
    # Tạo DataFrame
    df = pd.DataFrame(speaker_data)
    
    # Lưu vào file CSV
    output_file = "speaker_database.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nĐã tạo file {output_file} với {len(df)} speaker")
    print("\nMột số ví dụ:")
    print(df.head(10).to_string(index=False))
    
    return df

if __name__ == "__main__":
    create_speaker_database()
