#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để đọc file JSON và tải tất cả các file audio từ các URL trong đó
"""

import json
import os
import sys
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


def download_audio_from_json(json_file="get_audio.json", output_folder="new_audio_test"):
    """
    Đọc file JSON và tải tất cả các file audio từ các URL trong đó
    
    Args:
        json_file: Đường dẫn đến file JSON
        output_folder: Thư mục để lưu các file đã tải
    """
    console = Console()
    
    # Tạo thư mục output nếu chưa tồn tại
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Thư mục lưu: {output_path.absolute()}[/]\n")
    
    try:
        # Đọc file JSON
        console.print(f"[cyan]Đang đọc file JSON: {json_file}[/]")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract tất cả các URL từ rows
        urls = []
        sentences = []
        
        if 'rows' in data:
            for row_data in data['rows']:
                if 'row' in row_data and 'path' in row_data['row']:
                    path_list = row_data['row']['path']
                    if path_list and len(path_list) > 0:
                        url = path_list[0].get('src', '')
                        if url:
                            urls.append(url)
                            # Lấy sentence nếu có để đặt tên file
                            sentence = row_data['row'].get('sentence', '')
                            sentences.append(sentence)
        
        total = len(urls)
        if total == 0:
            console.print("[red]Không tìm thấy URL nào trong file JSON![/]")
            return False
        
        console.print(f"[green]Tìm thấy {total} file audio để tải[/]\n")
        
        # Tải từng file
        success_count = 0
        failed_count = 0
        
        with Progress(
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Đang tải...", total=total)
            
            for idx, (url, sentence) in enumerate(zip(urls, sentences), 1):
                try:
                    # Tạo tên file từ sentence hoặc index
                    if sentence:
                        # Làm sạch tên file (loại bỏ ký tự không hợp lệ)
                        safe_name = "".join(c for c in sentence if c.isalnum() or c in (' ', '-', '_', '.')).strip()
                        safe_name = safe_name.replace(' ', '_')[:100]  # Giới hạn độ dài
                        filename = f"{idx:03d}_{safe_name}.wav"
                    else:
                        filename = f"{idx:03d}_audio.wav"
                    
                    file_path = output_path / filename
                    
                    # Tải file
                    progress.update(task, description=f"[cyan]Đang tải {idx}/{total}: {filename[:50]}...")
                    
                    response = requests.get(url, stream=True, timeout=60)
                    response.raise_for_status()
                    
                    # Lưu file
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    success_count += 1
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    failed_count += 1
                    console.print(f"[red]Lỗi khi tải file {idx}: {str(e)}[/]")
                    progress.update(task, advance=1)
                    continue
        
        # Tóm tắt
        console.print(f"\n[green]✓ Hoàn thành![/]")
        console.print(f"[green]Thành công: {success_count}/{total}[/]")
        if failed_count > 0:
            console.print(f"[yellow]Thất bại: {failed_count}/{total}[/]")
        console.print(f"[cyan]File đã được lưu trong: {output_path.absolute()}[/]")
        
        return success_count > 0
        
    except FileNotFoundError:
        console.print(f"[red]Lỗi: Không tìm thấy file {json_file}[/]")
        return False
    except json.JSONDecodeError as e:
        console.print(f"[red]Lỗi: Không thể parse file JSON: {str(e)}[/]")
        return False
    except Exception as e:
        console.print(f"[red]Lỗi: {str(e)}[/]")
        return False


def main():
    """Hàm main"""
    console = Console()
    
    # Sử dụng tham số từ command line hoặc giá trị mặc định
    json_file = "get_audio.json"
    output_folder = "new_audio_test"
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    console.print(f"[yellow]Đọc file: {json_file}[/]")
    console.print(f"[yellow]Thư mục lưu: {output_folder}[/]\n")
    
    download_audio_from_json(json_file, output_folder)


if __name__ == "__main__":
    main()
