"""
Ablation Study Script (Phase 9)
===============================
Runs 2x2 grid: cls in {0.5, 1.2} x mosaic in {1.0, 0.0} across seeds {0, 1, 2} (12 runs).
Evaluates each run on VAL and TEST.

Outputs:
- results/ablation_runs.csv
- results/ablation_summary.csv
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = str(Path('braces_dataset_v3/data.yaml').resolve())
RESULTS_DIR = Path('results')

GRID_CLS = [0.5, 1.2]
GRID_MOSAIC = [1.0, 0.0]
SEEDS = [0, 1, 2]

def run_ablation_cell(cls_val, mos_val, seed_val, device):
    run_name = f"cls{cls_val}_mos{int(mos_val)}_s{seed_val}"
    print(f"\n==================================================")
    print(f"Ablation Run: {run_name} (cls={cls_val}, mosaic={mos_val}, seed={seed_val})")
    print(f"==================================================")
    
    project = 'runs/ablation'
    
    model = YOLO('yolov8n.pt')
    
    train_args = {
        'data': DATA_YAML,
        'epochs': 40,
        'imgsz': 512,
        'batch': 16,
        'project': project,
        'name': run_name,
        'exist_ok': True,
        'device': device,
        'patience': 10,
        'save': True,
        'plots': True,
        'verbose': False,
        'seed': seed_val,
        'deterministic': True,
        'lr0': 0.001,
        'lrf': 0.01,
        'optimizer': 'AdamW',
        'weight_decay': 0.0005,
        'box': 7.5,
        'cls': cls_val,
        'dfl': 1.5,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': mos_val,
        'mixup': 0.1,
        'copy_paste': 0.3,
        'close_mosaic': 10,
        'amp': True,
        'cache': True,
        'workers': 8
    }
    
    best_weights = Path('runs/detect/runs/ablation') / run_name / 'weights' / 'best.pt'
    if not best_weights.exists():
        results = model.train(**train_args)
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
            best_weights = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
            
    eval_model = YOLO(str(best_weights))
    
    # Evaluate on VAL
    val_res = eval_model.val(data=DATA_YAML, split='val', imgsz=512, conf=0.25, iou=0.50, verbose=False)
    # Evaluate on TEST
    test_res = eval_model.val(data=DATA_YAML, split='test', imgsz=512, conf=0.25, iou=0.50, verbose=False)
    
    return {
        'run_name': run_name,
        'cls': cls_val,
        'mosaic': mos_val,
        'seed': seed_val,
        'val_precision': round(float(val_res.box.mp), 4),
        'val_recall': round(float(val_res.box.mr), 4),
        'val_map50': round(float(val_res.box.map50), 4),
        'val_map50_95': round(float(val_res.box.map), 4),
        'test_precision': round(float(test_res.box.mp), 4),
        'test_recall': round(float(test_res.box.mr), 4),
        'test_map50': round(float(test_res.box.map50), 4),
        'test_map50_95': round(float(test_res.box.map), 4)
    }

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else ('0' if torch.cuda.is_available() else 'cpu')

    results_path = RESULTS_DIR / 'ablation_runs.csv'
    all_runs = []
    if results_path.exists():
        existing = pd.read_csv(results_path)
        if not existing.empty:
            all_runs = existing.to_dict(orient='records')

    completed = {row['run_name'] for row in all_runs}

    for cls_val in GRID_CLS:
        for mos_val in GRID_MOSAIC:
            for seed_val in SEEDS:
                run_name = f"cls{cls_val}_mos{int(mos_val)}_s{seed_val}"
                if run_name in completed:
                    continue

                row = run_ablation_cell(cls_val, mos_val, seed_val, device)
                all_runs.append(row)
                pd.DataFrame(all_runs).to_csv(results_path, index=False)

    df_runs = pd.DataFrame(all_runs)
    if df_runs.empty:
        raise RuntimeError('No ablation results were produced.')

    # Compute summary (mean and SD per config)
    summary_rows = []
    grouped = df_runs.groupby(['cls', 'mosaic'])

    for (cls_val, mos_val), group in grouped:
        summary_rows.append({
            'cls': cls_val,
            'mosaic': mos_val,
            'close_mosaic_epochs': 10,
            'val_map50_mean': round(float(group['val_map50'].mean()), 4),
            'val_map50_sd': round(float(group['val_map50'].std()), 4),
            'val_map50_95_mean': round(float(group['val_map50_95'].mean()), 4),
            'val_map50_95_sd': round(float(group['val_map50_95'].std()), 4),
            'test_map50_mean': round(float(group['test_map50'].mean()), 4),
            'test_map50_sd': round(float(group['test_map50'].std()), 4),
            'test_map50_95_mean': round(float(group['test_map50_95'].mean()), 4),
            'test_map50_95_sd': round(float(group['test_map50_95'].std()), 4)
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(RESULTS_DIR / 'ablation_summary.csv', index=False)
    print(f"\nPhase 9 complete! Saved ablation_runs.csv and ablation_summary.csv to {RESULTS_DIR}")

if __name__ == '__main__':
    main()
