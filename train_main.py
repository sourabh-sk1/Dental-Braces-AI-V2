"""
Train Main Model Script (Phase 3)
=================================
Trains YOLOv8n on braces_dataset_v3/data.yaml with exact baseline parameters.
Saves main artifacts to results/main/
"""

import os
import sys
import time
import json
import shutil
import platform
import psutil
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    start_time = time.time()
    
    # Target dataset and run name
    data_yaml = str(Path('braces_dataset_v3/data.yaml').resolve())
    project = 'runs'
    name = 'main'
    
    device = 'mps' if torch.backends.mps.is_available() else ('0' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Starting Main Model Training (YOLOv8n)")
    print(f"Device: {device}")
    print(f"Data YAML: {data_yaml}")
    print("=" * 60)
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=data_yaml,
        model='yolov8n.pt',
        epochs=40,
        imgsz=512,
        batch=16,
        project=project,
        name=name,
        exist_ok=True,
        device=device,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        seed=42,
        deterministic=True,
        # Learning rate
        lr0=0.001,
        lrf=0.01,
        optimizer='AdamW',
        weight_decay=0.0005,
        # Loss weights
        box=7.5,
        cls=1.2,
        dfl=1.5,
        # Augmentation
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        close_mosaic=10,
        amp=True
    )
    
    elapsed_sec = round(time.time() - start_time, 2)
    print(f"\nTraining completed in {elapsed_sec} seconds.")
    
    # Save artifacts to results/main/
    out_dir = Path('results/main')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_dir = Path(project) / name
    
    for f_name in ['args.yaml', 'results.csv', 'results.png']:
        src = run_dir / f_name
        if src.exists():
            shutil.copy2(src, out_dir / f_name)
            
    best_pt = run_dir / 'weights' / 'best.pt'
    if best_pt.exists():
        shutil.copy2(best_pt, out_dir / 'best.pt')
        
    cpu_model = platform.processor() or platform.machine()
    try:
        import subprocess
        cpu_model = subprocess.check_output('sysctl -n machdep.cpu.brand_string', shell=True).decode().strip()
    except Exception:
        pass
        
    info = {
        'training_time_seconds': elapsed_sec,
        'hardware': {
            'os': f"{platform.system()} {platform.release()}",
            'cpu': cpu_model,
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'device': device
        },
        'hyperparameters': {
            'model': 'yolov8n.pt',
            'epochs': 40,
            'imgsz': 512,
            'batch': 16,
            'cls': 1.2,
            'box': 7.5,
            'dfl': 1.5,
            'mosaic': 1.0,
            'seed': 42
        }
    }
    
    with open(out_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
        
    print(f"Artifacts successfully saved to {out_dir}")

if __name__ == '__main__':
    main()
