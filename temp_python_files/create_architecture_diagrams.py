#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViSpeech Architecture Visualization
Tạo biểu đồ kiến trúc hệ thống và luồng hoạt động
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Cài đặt font cho tiếng Việt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

def create_architecture_diagram():
    """Tạo biểu đồ kiến trúc tổng quan"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'ViSpeech - Voice Comparison System Architecture', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Main phases
    phases = [
        {'name': 'Training Phase', 'pos': (1.5, 9), 'color': '#FF6B6B', 'width': 2, 'height': 1.5},
        {'name': 'Inference Phase', 'pos': (5, 9), 'color': '#4ECDC4', 'width': 2, 'height': 1.5},
        {'name': 'Testing Phase', 'pos': (8.5, 9), 'color': '#45B7D1', 'width': 2, 'height': 1.5}
    ]
    
    for phase in phases:
        rect = FancyBboxPatch(
            (phase['pos'][0] - phase['width']/2, phase['pos'][1] - phase['height']/2),
            phase['width'], phase['height'],
            boxstyle="round,pad=0.1",
            facecolor=phase['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(phase['pos'][0], phase['pos'][1], phase['name'], 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Training components
    training_components = [
        'Raw Audio Files\n(trainset/)',
        'Feature Extraction\n(Librosa Pipeline)',
        'Super Metadata CSV\n(15+ features)',
        'Model Training\n(K-NN + RandomForest)'
    ]
    
    training_y_positions = [7.5, 6.5, 5.5, 4.5]
    for i, (comp, y_pos) in enumerate(zip(training_components, training_y_positions)):
        rect = FancyBboxPatch(
            (0.5, y_pos - 0.3), 2, 0.6,
            boxstyle="round,pad=0.05",
            facecolor='#FFE5E5',
            edgecolor='#FF6B6B',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(1.5, y_pos, comp, ha='center', va='center', fontsize=10)
        
        # Arrow
        if i < len(training_components) - 1:
            arrow = ConnectionPatch(
                (1.5, y_pos - 0.3), (1.5, training_y_positions[i+1] + 0.3),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="#FF6B6B"
            )
            ax.add_patch(arrow)
    
    # Inference components
    inference_components = [
        'Input Audio\n(Upload/Record)',
        'Audio Preprocessing\n(Auto-cut 20s)',
        'Feature Extraction\n(Same pipeline)',
        'Regional Detection\n(RandomForest)',
        'Voice Comparison\n(K-NN Search)',
        'Results Display\n+ JSON Export'
    ]
    
    inference_y_positions = [7.5, 6.5, 5.5, 4.5, 3.5, 2.5]
    for i, (comp, y_pos) in enumerate(zip(inference_components, inference_y_positions)):
        rect = FancyBboxPatch(
            (4, y_pos - 0.3), 2, 0.6,
            boxstyle="round,pad=0.05",
            facecolor='#E5F9F6',
            edgecolor='#4ECDC4',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(5, y_pos, comp, ha='center', va='center', fontsize=10)
        
        # Arrow
        if i < len(inference_components) - 1:
            arrow = ConnectionPatch(
                (5, y_pos - 0.3), (5, inference_y_positions[i+1] + 0.3),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="#4ECDC4"
            )
            ax.add_patch(arrow)
    
    # Testing components
    testing_components = [
        'Algorithm Testing\n(12 ML algorithms)',
        'Performance Analysis\n(Accuracy vs Time)',
        'Report Generation\n(Charts + Analysis)',
        'Model Comparison\n(Strengths/Weaknesses)'
    ]
    
    testing_y_positions = [7.5, 6.5, 5.5, 4.5]
    for i, (comp, y_pos) in enumerate(zip(testing_components, testing_y_positions)):
        rect = FancyBboxPatch(
            (7.5, y_pos - 0.3), 2, 0.6,
            boxstyle="round,pad=0.05",
            facecolor='#E5F4FF',
            edgecolor='#45B7D1',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(8.5, y_pos, comp, ha='center', va='center', fontsize=10)
        
        # Arrow
        if i < len(testing_components) - 1:
            arrow = ConnectionPatch(
                (8.5, y_pos - 0.3), (8.5, testing_y_positions[i+1] + 0.3),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="#45B7D1"
            )
            ax.add_patch(arrow)
    
    # Data flow arrows between phases
    # Training to Inference
    arrow1 = ConnectionPatch(
        (2.5, 8.2), (4.5, 8.2),
        "data", "data",
        arrowstyle="->", shrinkA=5, shrinkB=5,
        mutation_scale=20, fc="black", linewidth=2
    )
    ax.add_patch(arrow1)
    ax.text(3.5, 8.4, 'Trained Models', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Inference to Testing
    arrow2 = ConnectionPatch(
        (6, 8.2), (7.5, 8.2),
        "data", "data",
        arrowstyle="->", shrinkA=5, shrinkB=5,
        mutation_scale=20, fc="black", linewidth=2
    )
    ax.add_patch(arrow2)
    ax.text(6.75, 8.4, 'Performance Data', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Legend
    legend_elements = [
        patches.Patch(color='#FF6B6B', label='Training Phase'),
        patches.Patch(color='#4ECDC4', label='Inference Phase'),
        patches.Patch(color='#45B7D1', label='Testing Phase')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    plt.tight_layout()
    plt.savefig('architecture_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_architecture_diagram():
    """Tạo biểu đồ kiến trúc mô hình"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'ViSpeech - Model Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Input layer
    input_rect = FancyBboxPatch(
        (1, 8), 2, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='#E8F5E8',
        edgecolor='#4CAF50',
        linewidth=2
    )
    ax.add_patch(input_rect)
    ax.text(2, 8.4, 'Audio Input\n(MP3/WAV)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Feature extraction pipeline
    feature_rect = FancyBboxPatch(
        (4, 7), 4, 1.5,
        boxstyle="round,pad=0.1",
        facecolor='#FFF3E0',
        edgecolor='#FF9800',
        linewidth=2
    )
    ax.add_patch(feature_rect)
    ax.text(6, 7.75, 'Feature Extraction Pipeline\n(Librosa-based)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Feature details
    features = [
        'Pitch (pyin)',
        'MFCC (1-5)',
        'Spectral Centroid',
        'Zero Crossing Rate',
        'RMS Energy',
        'Tempo',
        'Duration',
        'Loudness',
        'Spectral Bandwidth',
        'Spectral Flatness',
        'Harmonic-to-Noise Ratio'
    ]
    
    feature_y = 6.5
    for i, feature in enumerate(features):
        if i % 3 == 0 and i > 0:
            feature_y -= 0.3
        x_pos = 4.2 + (i % 3) * 1.2
        ax.text(x_pos, feature_y, f'• {feature}', ha='left', va='center', fontsize=9)
    
    # Preprocessing
    preprocess_rect = FancyBboxPatch(
        (9, 7), 2, 1.5,
        boxstyle="round,pad=0.1",
        facecolor='#F3E5F5',
        edgecolor='#9C27B0',
        linewidth=2
    )
    ax.add_patch(preprocess_rect)
    ax.text(10, 7.75, 'Preprocessing\n(StandardScaler)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Models
    models_y = 5
    
    # Regional Detection Model
    regional_rect = FancyBboxPatch(
        (1, models_y), 3, 1.2,
        boxstyle="round,pad=0.1",
        facecolor='#E1F5FE',
        edgecolor='#03A9F4',
        linewidth=2
    )
    ax.add_patch(regional_rect)
    ax.text(2.5, models_y + 0.6, 'Regional Detection\nRandomForestClassifier', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Voice Comparison Model
    voice_rect = FancyBboxPatch(
        (5, models_y), 3, 1.2,
        boxstyle="round,pad=0.1",
        facecolor='#FFF8E1',
        edgecolor='#FFC107',
        linewidth=2
    )
    ax.add_patch(voice_rect)
    ax.text(6.5, models_y + 0.6, 'Voice Comparison\nK-Nearest Neighbors', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Algorithm Testing Model
    testing_rect = FancyBboxPatch(
        (9, models_y), 3, 1.2,
        boxstyle="round,pad=0.1",
        facecolor='#FCE4EC',
        edgecolor='#E91E63',
        linewidth=2
    )
    ax.add_patch(testing_rect)
    ax.text(10.5, models_y + 0.6, 'Algorithm Testing\n(12 ML Algorithms)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Outputs
    outputs_y = 3
    
    # Regional Output
    regional_out_rect = FancyBboxPatch(
        (1, outputs_y), 3, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='#E8F5E8',
        edgecolor='#4CAF50',
        linewidth=2
    )
    ax.add_patch(regional_out_rect)
    ax.text(2.5, outputs_y + 0.4, 'Predicted Region\n+ Confidence', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Voice Output
    voice_out_rect = FancyBboxPatch(
        (5, outputs_y), 3, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='#E8F5E8',
        edgecolor='#4CAF50',
        linewidth=2
    )
    ax.add_patch(voice_out_rect)
    ax.text(6.5, outputs_y + 0.4, 'Top K Similar\nSpeakers + Scores', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Testing Output
    testing_out_rect = FancyBboxPatch(
        (9, outputs_y), 3, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='#E8F5E8',
        edgecolor='#4CAF50',
        linewidth=2
    )
    ax.add_patch(testing_out_rect)
    ax.text(10.5, outputs_y + 0.4, 'Performance\nReport + Charts', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        # Input to Feature Extraction
        ((2, 7.6), (4, 7.6)),
        # Feature Extraction to Preprocessing
        ((8, 7.6), (9, 7.6)),
        # Preprocessing to Models
        ((10, 6.4), (2.5, 5.6)),
        ((10, 6.4), (6.5, 5.6)),
        ((10, 6.4), (10.5, 5.6)),
        # Models to Outputs
        ((2.5, 4.8), (2.5, 3.8)),
        ((6.5, 4.8), (6.5, 3.8)),
        ((10.5, 4.8), (10.5, 3.8))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(
            start, end,
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=15, fc="black", linewidth=1.5
        )
        ax.add_patch(arrow)
    
    # Performance metrics
    metrics_text = """
Performance Metrics:
• Feature Extraction: ~2-3 hours (8,166 files)
• Regional Detection: ~50ms per query
• Voice Comparison: ~100ms per query
• Overall Accuracy: 85-90% (top-5 matches)
• Regional Accuracy: 80-85%
    """
    
    ax.text(6, 1.5, metrics_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F5F5F5', edgecolor='#CCCCCC'))
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_data_flow_diagram():
    """Tạo biểu đồ luồng dữ liệu"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(8, 7.5, 'ViSpeech - Data Flow Diagram', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Data sources
    sources = [
        {'name': 'Raw Audio\n(trainset/)', 'pos': (1, 6), 'color': '#FF6B6B'},
        {'name': 'Metadata\n(trainset.csv)', 'pos': (1, 4), 'color': '#4ECDC4'},
        {'name': 'Speaker DB\n(speaker_database.csv)', 'pos': (1, 2), 'color': '#45B7D1'}
    ]
    
    for source in sources:
        rect = FancyBboxPatch(
            (source['pos'][0] - 0.8, source['pos'][1] - 0.4),
            1.6, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=source['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(source['pos'][0], source['pos'][1], source['name'], 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Processing steps
    steps = [
        {'name': 'Feature\nExtraction', 'pos': (4, 5), 'color': '#FFE5E5'},
        {'name': 'Data\nPreprocessing', 'pos': (7, 5), 'color': '#E5F9F6'},
        {'name': 'Model\nTraining', 'pos': (10, 5), 'color': '#E5F4FF'},
        {'name': 'Regional\nDetection', 'pos': (13, 5), 'color': '#FFF3E0'},
        {'name': 'Voice\nComparison', 'pos': (13, 3), 'color': '#F3E5F5'},
        {'name': 'Results\nDisplay', 'pos': (13, 1), 'color': '#E8F5E8'}
    ]
    
    for step in steps:
        rect = FancyBboxPatch(
            (step['pos'][0] - 0.8, step['pos'][1] - 0.4),
            1.6, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=step['color'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(step['pos'][0], step['pos'][1], step['name'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Data stores
    stores = [
        {'name': 'Super Metadata\nCSV Files', 'pos': (5.5, 3), 'color': '#FFF8E1'},
        {'name': 'Trained\nModels', 'pos': (8.5, 3), 'color': '#E1F5FE'},
        {'name': 'JSON\nResults', 'pos': (11.5, 3), 'color': '#FCE4EC'}
    ]
    
    for store in stores:
        rect = FancyBboxPatch(
            (store['pos'][0] - 0.8, store['pos'][1] - 0.4),
            1.6, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=store['color'],
            edgecolor='black',
            linewidth=1,
            linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(store['pos'][0], store['pos'][1], store['name'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Flow arrows
    flows = [
        # From sources to processing
        ((1.8, 6), (3.2, 5.4)),
        ((1.8, 4), (3.2, 4.6)),
        ((1.8, 2), (3.2, 4.6)),
        
        # Processing flow
        ((4.8, 5), (6.2, 5)),
        ((7.8, 5), (9.2, 5)),
        
        # To data stores
        ((10.8, 5), (11.2, 3.4)),
        ((10.8, 5), (8.8, 3.4)),
        ((10.8, 5), (5.8, 3.4)),
        
        # From stores to models
        ((6.3, 2.6), (12.2, 5.4)),
        ((9.3, 2.6), (12.2, 5.4)),
        ((12.3, 2.6), (12.2, 3.4)),
        
        # Model flow
        ((13.8, 4.6), (13.8, 3.4)),
        ((13.8, 2.6), (13.8, 1.4))
    ]
    
    for start, end in flows:
        arrow = ConnectionPatch(
            start, end,
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=15, fc="black", linewidth=1.5
        )
        ax.add_patch(arrow)
    
    # Labels for flows
    flow_labels = [
        ('Audio Files', (2.5, 5.7)),
        ('Metadata', (2.5, 4.3)),
        ('Speaker Info', (2.5, 3.3)),
        ('Features', (5.5, 5.2)),
        ('Processed Data', (8.5, 5.2)),
        ('Trained Models', (11.5, 4.2)),
        ('Predictions', (13.5, 4)),
        ('Results', (13.5, 2.2))
    ]
    
    for label, pos in flow_labels:
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Tạo tất cả biểu đồ kiến trúc"""
    print("Tao bieu do kien truc he thong...")
    
    # Tạo thư mục output
    import os
    os.makedirs("architecture_diagrams", exist_ok=True)
    
    # Thay đổi thư mục làm việc
    original_dir = os.getcwd()
    os.chdir("architecture_diagrams")
    
    try:
        # Tạo các biểu đồ
        create_architecture_diagram()
        print("✓ Tao architecture_overview.png")
        
        create_model_architecture_diagram()
        print("✓ Tao model_architecture.png")
        
        create_data_flow_diagram()
        print("✓ Tao data_flow_diagram.png")
        
        print("\nHoan thanh tao bieu do kien truc!")
        print("Cac file da tao trong thu muc 'architecture_diagrams/':")
        print("- architecture_overview.png")
        print("- model_architecture.png") 
        print("- data_flow_diagram.png")
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()
