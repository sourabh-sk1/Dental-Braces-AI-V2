"""
Evaluation Script for Main Model (Phase 4)
===========================================
Evaluates results/main/best.pt on TEST and VAL splits.
Generates evaluation_report.csv, confusion matrix, normalized matrix,
eval_settings.json, and high-res 300 DPI metrics.png.
"""

import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = str(Path('braces_dataset_v3/data.yaml').resolve())
MODEL_WEIGHTS = Path('results/main/best.pt')
RESULTS_DIR = Path('results')
MAIN_RESULTS_DIR = Path('results/main')

def generate_metrics_plot(results_csv_path: Path, out_path: Path):
    if not results_csv_path.exists():
        print(f"Warning: {results_csv_path} not found for plotting metrics.")
        return
        
    df = pd.read_csv(results_csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    epochs = df['epoch']
    
    # Best epoch based on mAP50-95
    best_epoch_idx = df['metrics/mAP50-95(B)'].idxmax()
    best_epoch = df.loc[best_epoch_idx, 'epoch']
    best_map = df.loc[best_epoch_idx, 'metrics/mAP50-95(B)']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
    fig.suptitle('YOLOv8n Training & Validation Metrics (Leak-Free Dataset)', fontsize=16, fontweight='bold')
    
    # Subplot 1: Box Loss
    ax = axes[0, 0]
    ax.plot(epochs, df['train/box_loss'], label='Train Box Loss', color='#1f77b4', lw=2)
    if 'val/box_loss' in df.columns:
        ax.plot(epochs, df['val/box_loss'], label='Val Box Loss', color='#ff7f0e', lw=2)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.set_title('Box Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Cls Loss
    ax = axes[0, 1]
    ax.plot(epochs, df['train/cls_loss'], label='Train Cls Loss', color='#1f77b4', lw=2)
    if 'val/cls_loss' in df.columns:
        ax.plot(epochs, df['val/cls_loss'], label='Val Cls Loss', color='#ff7f0e', lw=2)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.7)
    ax.set_title('Classification Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: DFL Loss
    ax = axes[0, 2]
    ax.plot(epochs, df['train/dfl_loss'], label='Train DFL Loss', color='#1f77b4', lw=2)
    if 'val/dfl_loss' in df.columns:
        ax.plot(epochs, df['val/dfl_loss'], label='Val DFL Loss', color='#ff7f0e', lw=2)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.7)
    ax.set_title('DFL Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Precision
    ax = axes[0, 3]
    ax.plot(epochs, df['metrics/precision(B)'], color='#2ca02c', lw=2, label='Precision')
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.7)
    ax.set_title('Precision (Val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 5: Recall
    ax = axes[1, 0]
    ax.plot(epochs, df['metrics/recall(B)'], color='#d62728', lw=2, label='Recall')
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.7)
    ax.set_title('Recall (Val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 6: mAP@50
    ax = axes[1, 1]
    ax.plot(epochs, df['metrics/mAP50(B)'], color='#9467bd', lw=2, label='mAP@50')
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.7)
    ax.set_title('mAP @ IoU=0.50 (Val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@50')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 7: mAP@50-95
    ax = axes[1, 2]
    ax.plot(epochs, df['metrics/mAP50-95(B)'], color='#8c564b', lw=2, label='mAP@50-95')
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Peak mAP: {best_map:.4f}')
    ax.set_title('mAP @ IoU=0.50:0.95 (Val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@50-95')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 8: Summary Info
    ax = axes[1, 3]
    ax.axis('off')
    summary_text = (
        f"Training Summary:\n"
        f"------------------\n"
        f"Total Epochs: {epochs.max()}\n"
        f"Best Epoch: {best_epoch}\n"
        f"Best mAP50-95: {best_map:.4f}\n"
        f"Final mAP50: {df['metrics/mAP50(B)'].iloc[-1]:.4f}\n"
        f"Final Precision: {df['metrics/precision(B)'].iloc[-1]:.4f}\n"
        f"Final Recall: {df['metrics/recall(B)'].iloc[-1]:.4f}\n"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace', va='center')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 300 DPI metrics plot to {out_path}")

def main():
    if not MODEL_WEIGHTS.exists():
        print(f"Error: {MODEL_WEIGHTS} does not exist. Train main model first!")
        return
        
    model = YOLO(str(MODEL_WEIGHTS))
    
    conf_thresh = 0.25
    iou_thresh = 0.50
    
    eval_rows = []
    
    for split in ['test', 'val']:
        report_split_name = 'valid' if split == 'val' else split
        print(f"\n--- Evaluating on {split.upper()} split (conf={conf_thresh}, iou={iou_thresh}) ---")
        val_res = model.val(
            data=DATA_YAML,
            split=split,
            imgsz=512,
            conf=conf_thresh,
            iou=iou_thresh,
            plots=True,
            project='runs/eval',
            name=f'main_{split}',
            exist_ok=True
        )
        
        # Parse results
        box_metrics = val_res.box
        
        # Per class metrics
        p_classes = box_metrics.p
        r_classes = box_metrics.r
        ap50_classes = box_metrics.ap50
        ap_classes = box_metrics.ap
        
        dir_split_name = 'valid' if split == 'val' else split
        
        # Count instances per class from target images
        lbl_dir = Path('braces_dataset_v3') / dir_split_name / 'labels'
        c0_instances = 0
        c1_instances = 0
        img_count = len(list((Path('braces_dataset_v3') / dir_split_name / 'images').glob('*')))
        
        for lbl in lbl_dir.glob('*.txt'):
            with open(lbl, 'r') as f:
                for l in f:
                    parts = l.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id == 0:
                            c0_instances += 1
                        elif cls_id == 1:
                            c1_instances += 1
                            
        class_names = ['Correct Brace', 'Incorrect Brace']
        
        for idx, c_name in enumerate(class_names):
            inst = c0_instances if idx == 0 else c1_instances
            prec = float(p_classes[idx]) if len(p_classes) > idx else 0.0
            rec = float(r_classes[idx]) if len(r_classes) > idx else 0.0
            map50_c = float(ap50_classes[idx]) if len(ap50_classes) > idx else 0.0
            map_c = float(ap_classes[idx]) if len(ap_classes) > idx else 0.0
            
            eval_rows.append({
                'split': report_split_name,
                'class': c_name,
                'images': img_count,
                'instances': inst,
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'map50': round(map50_c, 4),
                'map50_95': round(map_c, 4)
            })
            
        # Overall row
        eval_rows.append({
            'split': report_split_name,
            'class': 'all',
            'images': img_count,
            'instances': c0_instances + c1_instances,
            'precision': round(float(box_metrics.mp), 4),
            'recall': round(float(box_metrics.mr), 4),
            'map50': round(float(box_metrics.map50), 4),
            'map50_95': round(float(box_metrics.map), 4)
        })
        
        if split == 'test':
            # Copy confusion matrices
            eval_dir = Path('runs/eval') / f'main_{split}'
            for cm_name in ['confusion_matrix.png', 'confusion_matrix_normalized.png']:
                src_cm = eval_dir / cm_name
                if src_cm.exists():
                    shutil.copy2(src_cm, MAIN_RESULTS_DIR / cm_name)
                    if cm_name == 'confusion_matrix.png':
                        shutil.copy2(src_cm, Path('confusion_matrix.png'))
                        
    # Write evaluation_report.csv
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(RESULTS_DIR / 'evaluation_report.csv', index=False)
    df_eval.to_csv(MAIN_RESULTS_DIR / 'evaluation_report.csv', index=False)
    df_eval.to_csv(Path('evaluation_report.csv'), index=False)
    print(f"\nWritten evaluation_report.csv to results and root directory.")
    
    # Save eval settings info
    eval_settings = {
        'conf_threshold': conf_thresh,
        'iou_threshold': iou_thresh,
        'imgsz': 512,
        'model_weights': str(MODEL_WEIGHTS)
    }
    with open(MAIN_RESULTS_DIR / 'eval_settings.json', 'w') as f:
        json.dump(eval_settings, f, indent=2)
        
    # Generate 300 DPI metrics plot
    csv_src = MAIN_RESULTS_DIR / 'results.csv'
    if csv_src.exists():
        generate_metrics_plot(csv_src, MAIN_RESULTS_DIR / 'metrics.png')
        generate_metrics_plot(csv_src, Path('metrics.png'))

if __name__ == '__main__':
    main()
