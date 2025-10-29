#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViSpeech - Voice Comparison Desktop App (Final Version)
Ứng dụng desktop để so sánh giọng nói với tính năng replay và ghi âm
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import librosa
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import threading
import warnings
import pygame
import sounddevice as sd
import soundfile as sf
import tempfile
import subprocess
import json
import sys
warnings.filterwarnings('ignore')

class VoiceComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ViSpeech - Voice Comparison (Final)")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Khởi tạo VoiceComparisonKNN
        self.voice_comparison = None
        self.load_voice_comparison()
        
        # Khởi tạo pygame cho audio playback
        pygame.mixer.init()
        
        # Thiết lập callback khi audio kết thúc
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        
        # Bind event để reset trạng thái nút khi audio kết thúc
        self.root.bind('<KeyPress>', self._check_audio_end)
        
        # Timer để kiểm tra trạng thái audio định kỳ
        self.root.after(1000, self._check_audio_status)
        
        # Biến lưu trữ
        self.current_audio_features = None
        self.current_audio_path = None
        self.is_recording = False
        self.features_json_file = "current_audio_features.json"
        self.is_playing = False
        
        # Tạo giao diện
        self.create_widgets()
        
    def load_voice_comparison(self):
        """Load VoiceComparisonKNN"""
        try:
            self.voice_comparison = VoiceComparisonKNN()
            if self.voice_comparison.is_ready:
                print("VoiceComparisonKNN loaded successfully")
            else:
                print("VoiceComparisonKNN not ready - training required")
        except Exception as e:
            print(f"Error loading VoiceComparisonKNN: {e}")
            self.voice_comparison = None
    
    def create_widgets(self):
        """Tạo giao diện"""
        # Header
        header_frame = tk.Frame(self.root, bg='#667eea', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="ViSpeech - Voice Comparison (Final)", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#667eea')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame, text="So sanh va tim kiem giong noi tuong tu", 
                                 font=('Arial', 10), fg='white', bg='#667eea')
        subtitle_label.pack()
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # File selection
        file_frame = tk.LabelFrame(main_frame, text="Chon File Audio (Tu dong cat 20s dau)", 
                                  font=('Arial', 12, 'bold'), bg='#f0f0f0')
        file_frame.pack(fill='x', pady=10)
        
        self.file_path_var = tk.StringVar()
        file_entry = tk.Entry(file_frame, textvariable=self.file_path_var, 
                             font=('Arial', 10), state='readonly')
        file_entry.pack(side='left', fill='x', expand=True, padx=10, pady=10)
        
        browse_btn = tk.Button(file_frame, text="Chon File", command=self.browse_file,
                               bg='#667eea', fg='white', font=('Arial', 10, 'bold'))
        browse_btn.pack(side='right', padx=5, pady=10)
        
        self.record_btn = tk.Button(file_frame, text="Ghi am", command=self.start_recording,
                               bg='#dc3545', fg='white', font=('Arial', 10, 'bold'))
        self.record_btn.pack(side='right', padx=5, pady=10)
        
        # Control buttons
        control_frame = tk.Frame(main_frame, bg='#f0f0f0')
        control_frame.pack(fill='x', pady=10)
        
        compare_btn = tk.Button(control_frame, text="Kiem tra Giong noi", 
                               command=self.compare_voice, bg='#28a745', fg='white',
                               font=('Arial', 12, 'bold'), height=2)
        compare_btn.pack(side='left', fill='x', expand=True, padx=5)
        
        training_btn = tk.Button(control_frame, text="Training Du lieu", 
                                command=self.run_training, bg='#ffc107', fg='black',
                                font=('Arial', 10, 'bold'))
        training_btn.pack(side='right', padx=5)
        
        # Audio control buttons
        audio_control_frame = tk.Frame(control_frame, bg='#f0f0f0')
        audio_control_frame.pack(side='right', padx=5)
        
        self.replay_btn = tk.Button(audio_control_frame, text="▶️ Replay", 
                                   command=self.replay_current_audio, bg='#28a745', fg='white',
                                   font=('Arial', 9, 'bold'))
        self.replay_btn.pack(side='left', padx=2)
        
        self.pause_btn = tk.Button(audio_control_frame, text="⏸️ Pause", 
                                 command=self.pause_current_audio, bg='#dc3545', fg='white',
                                 font=('Arial', 9, 'bold'), state='disabled')
        self.pause_btn.pack(side='left', padx=2)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="San sang")
        progress_label = tk.Label(main_frame, textvariable=self.progress_var, 
                                font=('Arial', 10), bg='#f0f0f0')
        progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=5)
        
        # Audio Features Display
        features_frame = tk.LabelFrame(main_frame, text="Thuoc tinh Audio", 
                                      font=('Arial', 12, 'bold'), bg='#f0f0f0')
        features_frame.pack(fill='x', pady=10)
        
        self.features_text = tk.Text(features_frame, height=6, font=('Courier', 9), 
                                    bg='#f8f9fa', state='disabled')
        features_scrollbar = ttk.Scrollbar(features_frame, orient='vertical', 
                                          command=self.features_text.yview)
        self.features_text.configure(yscrollcommand=features_scrollbar.set)
        
        self.features_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        features_scrollbar.pack(side='right', fill='y', pady=10)
        
        # Results
        results_frame = tk.LabelFrame(main_frame, text="Ket qua", 
                                     font=('Arial', 12, 'bold'), bg='#f0f0f0')
        results_frame.pack(fill='both', expand=True, pady=10)
        
        # Treeview for results
        columns = ('Rank', 'Speaker', 'Similarity', 'Audio File', 'Dialect', 'Replay')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        self.results_tree.heading('Rank', text='Thu hang')
        self.results_tree.heading('Speaker', text='Speaker')
        self.results_tree.heading('Similarity', text='Do tuong tu (%)')
        self.results_tree.heading('Audio File', text='File Audio')
        self.results_tree.heading('Dialect', text='Phuong ngữ')
        self.results_tree.heading('Replay', text='Replay')
        
        self.results_tree.column('Rank', width=60)
        self.results_tree.column('Speaker', width=120)
        self.results_tree.column('Similarity', width=100)
        self.results_tree.column('Audio File', width=150)
        self.results_tree.column('Dialect', width=80)
        self.results_tree.column('Replay', width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)
        
        # Bind click event for Replay column
        self.results_tree.bind('<Button-1>', self.on_result_click)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#e0e0e0', height=30)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_var = tk.StringVar(value="San sang")
        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                               font=('Arial', 9), bg='#e0e0e0')
        status_label.pack(side='left', padx=10, pady=5)
        
        # Check system status
        self.check_system_status()
    
    def check_system_status(self):
        """Kiểm tra trạng thái hệ thống"""
        if self.voice_comparison is None:
            self.status_var.set("He thong chua san sang")
            return
        
        if not self.voice_comparison.is_ready:
            self.status_var.set("Can training du lieu")
            return
        
        # Kiểm tra dữ liệu training
        sample_features = self.voice_comparison.df[self.voice_comparison.feature_columns].iloc[0]
        if all(val == 0 for val in sample_features):
            self.status_var.set("Du lieu chua duoc training")
            return
        
        self.status_var.set("He thong san sang")
    
    def run_training(self):
        """Chạy training dữ liệu"""
        try:
            self.progress_var.set("Dang training du lieu...")
            self.progress_bar.start()
            self.status_var.set("Dang training...")
            
            # Chạy training trong thread riêng
            thread = threading.Thread(target=self._training_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.progress_bar.stop()
            self.progress_var.set("Loi training")
            messagebox.showerror("Loi", f"Khong the chay training: {str(e)}")
    
    def _training_thread(self):
        """Thread để chạy training"""
        try:
            # Import và chạy training
            from train_audio_features import AudioFeatureTrainer
            
            trainer = AudioFeatureTrainer()
            trainer.run_training()
            
            # Reload voice comparison
            self.root.after(0, lambda: self.load_voice_comparison())
            
            # Cập nhật UI
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.progress_var.set("Training hoan thanh"))
            self.root.after(0, lambda: self.status_var.set("Training hoan thanh"))
            self.root.after(0, lambda: messagebox.showinfo("Thanh cong", "Training hoan thanh! He thong da san sang."))
            
        except Exception as e:
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.progress_var.set("Loi training"))
            self.root.after(0, lambda: messagebox.showerror("Loi", f"Loi khi training: {str(e)}"))
    
    def browse_file(self):
        """Chọn file audio"""
        file_types = [
            ('Audio files (auto-cut to 20s)', '*.mp3 *.wav *.m4a *.flac'),
            ('MP3 files', '*.mp3'),
            ('WAV files', '*.wav'),
            ('M4A files', '*.m4a'),
            ('FLAC files', '*.flac'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Chọn file audio",
            filetypes=file_types
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.current_audio_path = filename
            # Kiểm tra độ dài file
            try:
                y, sr = librosa.load(filename)
                duration = len(y) / sr
                if duration > 20:
                    self.status_var.set(f"File dai {duration:.1f}s - Se tu dong cat lay 20s dau")
                else:
                    self.status_var.set(f"Da chon: {os.path.basename(filename)} ({duration:.1f}s)")
            except:
                self.status_var.set(f"Da chon: {os.path.basename(filename)}")
    
    def start_recording(self):
        """Bắt đầu ghi âm"""
        if self.is_recording:
            messagebox.showwarning("Canh bao", "Dang ghi am roi!")
            return
            
        try:
            self.is_recording = True
            self.record_btn.config(text="Dang ghi am...", bg='#ffc107', state='disabled')
            
            # Cài đặt ghi âm
            duration = 10  # Ghi âm 10 giây
            sample_rate = 44100
            
            self.progress_var.set("Dang ghi am... (10 giay)")
            self.progress_bar.start()
            self.status_var.set("Dang ghi am... Hay noi!")
            
            # Chạy ghi âm trong thread riêng
            thread = threading.Thread(target=self._recording_thread, args=(duration, sample_rate))
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.is_recording = False
            self.record_btn.config(text="Ghi am", bg='#dc3545', state='normal')
            self.progress_bar.stop()
            self.progress_var.set("Loi ghi am")
            messagebox.showerror("Loi", f"Khong the ghi am: {str(e)}")
    
    def _recording_thread(self, duration, sample_rate):
        """Thread ghi âm"""
        try:
            # Ghi âm
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()  # Chờ ghi âm hoàn thành
            
            # Lưu file tạm
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, recording, sample_rate)
            
            # Cập nhật UI
            self.root.after(0, lambda: self.file_path_var.set(temp_file.name))
            self.root.after(0, lambda: setattr(self, 'current_audio_path', temp_file.name))
            
            # Kiểm tra độ dài
            y, sr = librosa.load(temp_file.name)
            duration_actual = len(y) / sr
            
            # Cập nhật UI
            self.root.after(0, lambda: self.status_var.set(f"Da ghi am: {duration_actual:.1f}s"))
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.progress_var.set("Ghi am hoan thanh"))
            self.root.after(0, lambda: self.record_btn.config(text="Ghi am", bg='#dc3545', state='normal'))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Loi", f"Khong the ghi am: {str(e)}"))
        finally:
            self.is_recording = False
    
    def replay_current_audio(self):
        """Replay audio hiện tại"""
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            messagebox.showwarning("Canh bao", "Khong co audio de replay!")
            return
        
        try:
            # Dừng audio hiện tại nếu đang phát
            pygame.mixer.music.stop()
            
            # Phát audio mới
            pygame.mixer.music.load(self.current_audio_path)
            pygame.mixer.music.play()
            
            self.is_playing = True
            self.status_var.set("Dang phat audio...")
            
            # Cập nhật trạng thái nút
            self.replay_btn.config(text="▶️ Playing", bg='#20c997')
            self.pause_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the phat audio: {str(e)}")
    
    def _check_audio_status(self):
        """Kiểm tra trạng thái audio định kỳ"""
        try:
            if self.is_playing and not pygame.mixer.music.get_busy():
                # Audio đã kết thúc
                self.is_playing = False
                self.status_var.set("Audio da ket thuc")
                
                # Reset trạng thái nút
                self.replay_btn.config(text="▶️ Replay", bg='#28a745')
                self.pause_btn.config(text="⏸️ Pause", bg='#dc3545', state='disabled')
                
        except:
            pass
        
        # Lên lịch kiểm tra lại sau 1 giây
        self.root.after(1000, self._check_audio_status)
    
    def pause_current_audio(self):
        """Pause/Resume audio hiện tại"""
        try:
            if self.is_playing and pygame.mixer.music.get_busy():
                # Đang phát -> Pause
                pygame.mixer.music.pause()
                self.is_playing = False
                self.status_var.set("Audio da tam dung")
                self.pause_btn.config(text="▶️ Resume", bg='#28a745')
                
            elif not self.is_playing and pygame.mixer.music.get_pos() > 0:
                # Đã pause -> Resume
                pygame.mixer.music.unpause()
                self.is_playing = True
                self.status_var.set("Dang phat audio...")
                self.pause_btn.config(text="⏸️ Pause", bg='#dc3545')
                
            else:
                # Không có audio để pause/resume
                messagebox.showwarning("Canh bao", "Khong co audio dang phat!")
                
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the pause/resume audio: {str(e)}")
    
    def on_result_click(self, event):
        """Xử lý click trên kết quả"""
        item = self.results_tree.identify('item', event.x, event.y)
        column = self.results_tree.identify('column', event.x, event.y)
        
        if item and column:
            # Kiểm tra nếu click vào cột Replay (cột 6)
            if column == '#6':  # Cột Replay
                values = self.results_tree.item(item, 'values')
                audio_name = values[3]  # Audio File column
                self.replay_result_audio(audio_name)
    
    def replay_result_audio(self, audio_name):
        """Replay audio từ kết quả bằng Windows Media Player"""
        audio_path = os.path.join("trainset", audio_name)
        if not os.path.exists(audio_path):
            messagebox.showwarning("Canh bao", f"Khong tim thay file: {audio_name}")
            return
        
        try:
            # Mở file bằng Windows Media Player
            subprocess.Popen(['start', audio_path], shell=True)
            self.status_var.set(f"Dang mo: {audio_name}")
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the mo file: {str(e)}")
    
    def compare_voice(self):
        """So sánh giọng nói"""
        if not self.file_path_var.get():
            messagebox.showerror("Loi", "Vui long chon file audio!")
            return
        
        if self.voice_comparison is None:
            messagebox.showerror("Loi", "He thong chua san sang!")
            return
        
        # Kiểm tra xem có cần training không
        if not self.voice_comparison.is_ready:
            response = messagebox.askyesno("Can training", 
                                          "Du lieu chua duoc training!\nBan co muon chay training ngay bay gio khong?")
            if response:
                self.run_training()
            return
        
        # Kiểm tra dữ liệu training
        sample_features = self.voice_comparison.df[self.voice_comparison.feature_columns].iloc[0]
        if all(val == 0 for val in sample_features):
            response = messagebox.askyesno("Can training", 
                                          "Du lieu chua duoc training!\nBan co muon chay training ngay bay gio khong?")
            if response:
                self.run_training()
            return
        
        # Chạy trong thread riêng để không block UI
        thread = threading.Thread(target=self._compare_voice_thread)
        thread.daemon = True
        thread.start()
    
    def _compare_voice_thread(self):
        """Thread để so sánh giọng nói"""
        try:
            # Cập nhật UI
            self.root.after(0, lambda: self.progress_var.set("Dang phan tich audio..."))
            self.root.after(0, lambda: self.progress_bar.start())
            
            # Bước 1: Trích xuất thuộc tính audio
            audio_path = self.file_path_var.get()
            features = self.voice_comparison.extract_audio_features(audio_path)
            
            if not features:
                self.root.after(0, lambda: messagebox.showerror("Loi", "Khong the trich xuat dac trung tu file audio!"))
                return
            
            # Kiểm tra lỗi khác (không còn kiểm tra duration)
            if isinstance(features, dict) and 'error' in features:
                error_msg = features['message']
                self.root.after(0, lambda: messagebox.showerror("Loi", error_msg))
                return
            
            # Bước 2: Lưu thuộc tính vào JSON
            self.current_audio_features = features
            self._save_features_to_json(features)
            
            # Bước 3: Hiển thị thuộc tính
            self.root.after(0, lambda: self._display_audio_features())
            
            # Bước 4: Thực hiện so sánh
            self.root.after(0, lambda: self.progress_var.set("Dang so sanh..."))
            results = self.voice_comparison.find_similar_voices(audio_path, k=10, features=features)
            
            # Kiểm tra kết quả có lỗi không
            if isinstance(results, dict) and 'error' in results:
                error_msg = results['message']
                self.root.after(0, lambda: messagebox.showerror("Loi", error_msg))
                return
            
            # Bước 5: Cập nhật UI với kết quả
            self.root.after(0, lambda: self._update_results(results))
            
        except Exception as e:
            error_msg = f"Loi khi so sanh: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Loi", error_msg))
        finally:
            # Dừng progress bar
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.progress_var.set("Hoan thanh"))
    
    def _save_features_to_json(self, features):
        """Lưu thuộc tính audio vào file JSON"""
        try:
            # Thêm thông tin metadata
            features_data = {
                "audio_file": self.file_path_var.get(),
                "timestamp": pd.Timestamp.now().isoformat(),
                "features": features
            }
            
            # Lưu vào file JSON
            with open(self.features_json_file, 'w', encoding='utf-8') as f:
                json.dump(features_data, f, indent=2, ensure_ascii=False)
            
            print(f"Da luu thuoc tinh vao: {self.features_json_file}")
            
        except Exception as e:
            print(f"Loi khi luu JSON: {e}")
    
    def _update_results(self, results):
        """Cập nhật kết quả"""
        # Xóa kết quả cũ
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if not results:
            messagebox.showerror("Loi", "Khong tim thay ket qua!")
            return
        
        # Hiển thị thuộc tính audio
        self._display_audio_features()
        
        # Thêm kết quả mới
        for result in results:
            self.results_tree.insert('', 'end', values=(
                result['rank'],
                result['speaker_name'],
                f"{result['similarity']:.2f}%",
                result['audio_name'],
                result['dialect'],
                "Click to play"
            ))
        
        self.status_var.set(f"Tim thay {len(results)} ket qua tuong tu")
    
    def _display_audio_features(self):
        """Hiển thị thuộc tính audio từ JSON"""
        if not self.current_audio_features:
            return
        
        self.features_text.config(state='normal')
        self.features_text.delete(1.0, tk.END)
        
        features_text = "THUOC TINH AUDIO DA PHAN TICH:\n"
        features_text += "=" * 50 + "\n\n"
        
        # Hiển thị thông tin file JSON
        features_text += f"File JSON: {self.features_json_file}\n"
        features_text += f"Audio file: {os.path.basename(self.file_path_var.get())}\n"
        features_text += f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Hiển thị JSON format
        features_text += "JSON FORMAT:\n"
        features_text += "-" * 30 + "\n"
        
        try:
            # Đọc và hiển thị JSON
            with open(self.features_json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Hiển thị JSON đẹp
            features_text += json.dumps(json_data, indent=2, ensure_ascii=False)
            
        except Exception as e:
            features_text += f"Loi doc JSON: {e}\n"
            
            # Fallback: hiển thị thuộc tính trực tiếp
            features_text += "\nTHUOC TINH TRUC TIEP:\n"
            features_text += "-" * 30 + "\n"
            
            important_features = {
                'pitch_mean': 'Pitch trung binh (Hz)',
                'pitch_std': 'Pitch do lech chuan (Hz)',
                'spectral_centroid_mean': 'Spectral centroid TB',
                'mfcc_1_mean': 'MFCC 1',
                'mfcc_2_mean': 'MFCC 2',
                'mfcc_3_mean': 'MFCC 3',
                'zcr_mean': 'Zero crossing rate',
                'rms_mean': 'RMS energy',
                'tempo': 'Tempo (BPM)',
                'duration': 'Do dai (giay)',
                'loudness': 'Loudness (dB)',
                'spectral_bandwidth_mean': 'Spectral bandwidth',
                'spectral_flatness_mean': 'Spectral flatness',
                'hnr': 'Harmonic-to-noise ratio'
            }
            
            for feature, description in important_features.items():
                if feature in self.current_audio_features:
                    value = self.current_audio_features[feature]
                    features_text += f"{description:25}: {value:10.4f}\n"
        
        self.features_text.insert(1.0, features_text)
        self.features_text.config(state='disabled')

class VoiceComparisonKNN:
    def __init__(self):
        self.super_metadata_file = "super_metadata/male_only_merged.csv"
        self.speaker_db_file = "speaker_database.csv"
        self.scaler = StandardScaler()
        self.knn_model = None
        self.feature_columns = None
        self.speaker_data = None
        self.df = None
        self.is_ready = False
        
        # Load dữ liệu
        self.load_data()
        
    def load_data(self):
        """Load dữ liệu từ super metadata và speaker database"""
        try:
            # Load super metadata
            if os.path.exists(self.super_metadata_file):
                self.df = pd.read_csv(self.super_metadata_file, encoding='utf-8')
                print(f"Da load {len(self.df)} records tu super metadata")
            else:
                print("Khong tim thay super metadata file!")
                return
            
            # Load speaker database
            if os.path.exists(self.speaker_db_file):
                self.speaker_data = pd.read_csv(self.speaker_db_file, encoding='utf-8')
                print(f"Da load {len(self.speaker_data)} speakers tu database")
            else:
                print("Khong tim thay speaker database!")
                return
            
            # Chuẩn bị features cho K-NN
            self.prepare_features()
            
        except Exception as e:
            print(f"Loi khi load du lieu: {e}")
    
    def prepare_features(self):
        """Chuẩn bị features cho K-NN model"""
        try:
            # Chọn các features quan trọng
            self.feature_columns = [
                'pitch_mean', 'pitch_std', 'spectral_centroid_mean', 'spectral_centroid_std',
                'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean',
                'zcr_mean', 'rms_mean', 'tempo', 'loudness', 'duration',
                'spectral_bandwidth_mean', 'spectral_flatness_mean', 'hnr'
            ]
            
            # Lấy features từ dataframe
            X = self.df[self.feature_columns].fillna(0)
            
            # Chuẩn hóa features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train K-NN model
            self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
            self.knn_model.fit(X_scaled)
            
            self.is_ready = True
            print(f"Da train K-NN model voi {len(self.feature_columns)} features")
            
        except Exception as e:
            print(f"Loi khi chuan bi features: {e}")
            self.is_ready = False
    
    def extract_audio_features(self, audio_path):
        """Trích xuất đặc trưng âm thanh từ file audio"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path)
            
            # Kiểm tra độ dài audio và cắt nếu cần
            duration = len(y) / sr
            if duration > 20:
                print(f"File audio dai {duration:.1f}s - Tu dong cat lay 20s dau")
                # Cắt lấy 20 giây đầu
                y = y[:int(20 * sr)]
                duration = 20.0
            
            features = {}
            
            # 1. Pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            features['pitch_mean'] = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0
            features['pitch_std'] = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0.0
            
            # 2. Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # 3. MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(5):  # Chỉ lấy 5 MFCC đầu
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            
            # 5. RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = float(np.mean(rms))
            
            # 6. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # 7. Duration
            features['duration'] = float(len(y) / sr)
            
            # 8. Loudness
            features['loudness'] = float(20 * np.log10(np.mean(np.abs(y)) + 1e-10))
            
            # 9. Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # 10. Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            
            # 11. Harmonic-to-noise ratio
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            hnr = np.mean(20 * np.log10(np.abs(y_harmonic) / (np.abs(y_percussive) + 1e-10)))
            features['hnr'] = float(hnr) if not np.isnan(hnr) and not np.isinf(hnr) else 0.0
            
            return features
            
        except Exception as e:
            print(f"Loi khi trich xuat dac trung: {e}")
            return None
    
    def find_similar_voices(self, audio_path, k=10, features=None):
        """Tìm K giọng tương tự nhất sử dụng K-NN"""
        try:
            # Kiểm tra xem có dữ liệu thực không
            sample_features = self.df[self.feature_columns].iloc[0]
            if all(val == 0 for val in sample_features):
                return {
                    'error': 'training_required',
                    'message': 'Du lieu chua duoc training! Tat ca dac trung deu la 0. Hay chay training truoc.'
                }
            
            # Sử dụng thuộc tính đã được truyền vào hoặc trích xuất mới
            if features is None:
                features = self.extract_audio_features(audio_path)
                if not features:
                    return {
                        'error': 'extraction_failed',
                        'message': 'Khong the trich xuat dac trung tu file audio!'
                    }
            
            # Chuẩn bị vector features
            feature_vector = np.array([features.get(col, 0) for col in self.feature_columns])
            feature_vector = feature_vector.reshape(1, -1)
            
            # Chuẩn hóa features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Tìm K neighbors gần nhất
            distances, indices = self.knn_model.kneighbors(feature_vector_scaled, n_neighbors=k)
            
            # Tạo kết quả
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                # Tính similarity percentage (1 - distance) * 100
                similarity = max(0, (1 - dist) * 100)
                
                # Lấy thông tin từ dataframe
                row = self.df.iloc[idx]
                
                # Lấy tên speaker từ speaker database
                speaker_name = "Unknown"
                if self.speaker_data is not None:
                    speaker_info = self.speaker_data[self.speaker_data['speaker_id'] == row['speaker']]
                    if not speaker_info.empty:
                        speaker_name = speaker_info.iloc[0]['vietnamese_name']
                
                results.append({
                    'rank': i + 1,
                    'audio_name': row['audio_name'],
                    'speaker_id': row['speaker'],
                    'speaker_name': speaker_name,
                    'dialect': row['dialect'],
                    'similarity': round(similarity, 2),
                    'distance': round(dist, 4)
                })
            
            return results
            
        except Exception as e:
            print(f"Loi khi tim giong tuong tu: {e}")
            return {
                'error': 'processing_failed',
                'message': f'Loi xu ly: {str(e)}'
            }

def main():
    """Hàm chính"""
    print("ViSpeech - Voice Comparison Desktop App (Final)")
    print("=" * 60)
    
    root = tk.Tk()
    app = VoiceComparisonApp(root)
    
    print("Khoi dong ung dung...")
    root.mainloop()

if __name__ == "__main__":
    main()
