"""
Baseline Models Script (Phase 8)
================================
Trains and evaluates baseline models:
1. yolov8s.pt
2. yolov5nu.pt
3. yolov8n from scratch (yolov8n.yaml, pretrained=False)

Outputs results/baselines.csv
"""

import os
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = str(Path('braces_dataset_v3/data.yaml').resolve())
RESULTS_DIR = Path('results')
TEST_DIR = Path('braces_dataset_v3/test/images')

BASELINES = [
    {'name': 'yolov8s', 'init': 'yolov8s.pt'},
    {'name': 'yolov5nu', 'init': 'yolov5nu.pt'},
    {'name': 'yolov8n_scratch', 'init': 'yolov8n.yaml'}
]

def measure_cpu_latency(model, img_paths, n_warmup=10, n_runs=50):
    for i in range(n_warmup):
        p = img_paths[i % len(img_paths)]
        _ = model(str(p), device='cpu', conf=0.40, iou=0.45, imgsz=512, verbose=False)
        
    times = []
    for i in range(n_runs):
        p = img_paths[i % len(img_paths)]
        t0 = time.perf_counter()
        _ = model(str(p), device='cpu', conf=0.40, iou=0.45, imgsz=512, verbose=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return round(float(np.mean(times)), 2)

def train_and_eval_baseline(b_info, img_paths, device):
    name = f"baseline_{b_info['name']}"
    print(f"\n==================================================")
    print(f"Training Baseline: {name} (Init: {b_info['init']})")
    print(f"==================================================")
    
    if b_info['name'] == 'yolov8n_scratch':
        model = YOLO(b_info['init']) # initialized architecture from yaml
    else:
        model = YOLO(b_info['init'])
        
    project = 'runs/baselines'
    
    train_args = {
        'data': DATA_YAML,
        'epochs': 40,
        'imgsz': 512,
        'batch': 16,
        'project': project,
        'name': name,
        'exist_ok': True,
        'device': device,
        'patience': 10,
        'save': True,
        'plots': True,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'lr0': 0.001,
        'lrf': 0.01,
        'optimizer': 'AdamW',
        'weight_decay': 0.0005,
        'box': 7.5,
        'cls': 1.2,
        'dfl': 1.5,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.3,
        'close_mosaic': 10,
        'amp': True,
        'cache': True,
        'workers': 8
    }
    
    best_weights = Path('runs/detect/runs/baselines') / name / 'weights' / 'best.pt'
    if not best_weights.exists():
        model.train(**train_args)
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
            best_weights = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
            
    eval_model = YOLO(str(best_weights))
    
    print(f"Evaluating {name} on TEST split...")
    val_res = eval_model.val(data=DATA_YAML, split='test', imgsz=512, conf=0.25, iou=0.50, verbose=False)
    
    weight_size_mb = round(os.path.getsize(best_weights) / (1024 * 1024), 2)
    params_count = sum(p.numel() for p in eval_model.model.parameters())
    
    print(f"Measuring CPU latency for {name}...")
    cpu_latency = measure_cpu_latency(eval_model, img_paths)
    
    return {
        'model': b_info['name'],
        'precision': round(float(val_res.box.mp), 4),
        'recall': round(float(val_res.box.mr), 4),
        'map50': round(float(val_res.box.map50), 4),
        'map50_95': round(float(val_res.box.map), 4),
        'parameters': params_count,
        'weight_size_mb': weight_size_mb,
        'cpu_latency_ms': cpu_latency
    }

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else ('0' if torch.cuda.is_available() else 'cpu')
    img_paths = sorted(list(TEST_DIR.glob('*')))
    
    baseline_rows = []
    
    # Also include main model (YOLOv8n pretrained) in baselines table for complete comparison
    main_weights = Path('results/main/best.pt')
    if main_weights.exists():
        main_model = YOLO(str(main_weights))
        val_res = main_model.val(data=DATA_YAML, split='test', imgsz=512, conf=0.25, iou=0.50, verbose=False)
        weight_size_mb = round(os.path.getsize(main_weights) / (1024 * 1024), 2)
        params_count = sum(p.numel() for p in main_model.model.parameters())
        cpu_lat = measure_cpu_latency(main_model, img_paths)
        
        baseline_rows.append({
            'model': 'yolov8n (main pretrained)',
            'precision': round(float(val_res.box.mp), 4),
            'recall': round(float(val_res.box.mr), 4),
            'map50': round(float(val_res.box.map50), 4),
            'map50_95': round(float(val_res.box.map), 4),
            'parameters': params_count,
            'weight_size_mb': weight_size_mb,
            'cpu_latency_ms': cpu_lat
        })
        
    for b in BASELINES:
        row = train_and_eval_baseline(b, img_paths, device)
        baseline_rows.append(row)
        
    df_base = pd.DataFrame(baseline_rows)
    df_base.to_csv(RESULTS_DIR / 'baselines.csv', index=False)
    print(f"\nPhase 8 complete! Written {RESULTS_DIR / 'baselines.csv'}")

if __name__ == '__main__':
    main()
