#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy nhanh để so sánh 2 models
Chỉ cần chạy file này, các tham số đã được hardcode
"""

import os
import sys
import subprocess

if __name__ == '__main__':
    # Đường dẫn script chính
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, 'compare_models_with_without_delta.py')
    
    # Các tham số mặc định (có thể chỉnh sửa ở đây)
    cmd = [
        sys.executable,
        main_script,
        '--ckpt_no_delta', r'runs\change3d\20251031_224419_nodenta\best_change3d_dir.pth',
        '--ckpt_with_delta', r'runs\change3d\20251031_230655\best_change3d_dir.pth',
        '--test_dir', r'outputs\sum\dataset_new\val',
        '--batch_size', '128',
        '--workers', '0',
        '--output_dir', 'test'
    ]
    
    print("[INFO] Chạy so sánh models...")
    print(f"[INFO] Command: {' '.join(cmd)}")
    subprocess.run(cmd)

