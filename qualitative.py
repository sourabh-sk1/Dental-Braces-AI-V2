"""
Qualitative Examples Script (Phase 6)
=====================================
Generates visual qualitative candidate overlays and a contact sheet
in results/qualitative/ using test split images and tooth mapping.
"""

import sys
import os
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Import existing tooth mapping from teeth-braces-ai/utils/tooth_mapping.py
sys.path.append(str(Path('teeth-braces-ai').resolve()))
from utils.tooth_mapping import (
    get_tooth_from_position,
    TOOTH_SHORT_CODES,
    get_color_for_detection
)

TEST_DIR = Path('braces_dataset_v3/test')
MODEL_WEIGHTS = Path('results/main/best.pt')
OUT_DIR = Path('results/qualitative')

def main():
    if not MODEL_WEIGHTS.exists():
        print(f"Error: {MODEL_WEIGHTS} missing. Train main model first!")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_WEIGHTS))
    
    img_dir = TEST_DIR / 'images'
    lbl_dir = TEST_DIR / 'labels'
    img_paths = sorted(list(img_dir.glob('*')))
    
    cases = {
        'correct_only': [],
        'has_incorrect': [],
        'multi_bracket': [],
        'low_confidence': [],
        'failure_case': []
    }
    
    anomalies = []
    annotated_records = []
    
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        
        # Read GT
        gt_classes = []
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.append(int(parts[0]))
                        
        # Model inference at conf=0.25
        results = model(str(img_path), conf=0.25, iou=0.50, imgsz=512, verbose=False)[0]
        
        preds = []
        tooth_counts = {}
        
        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())
                
                # Tooth mapping
                xc_norm = ((x1 + x2) / 2) / img_w
                yc_norm = ((y1 + y2) / 2) / img_h
                w_norm = (x2 - x1) / img_w
                h_norm = (y2 - y1) / img_h
                
                norm_bbox = (xc_norm, yc_norm, w_norm, h_norm)
                tooth_key = get_tooth_from_position(norm_bbox, (img_h, img_w))
                tooth_short = TOOTH_SHORT_CODES.get(tooth_key, 'UNK')
                
                tooth_counts[tooth_short] = tooth_counts.get(tooth_short, 0) + 1
                
                preds.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'class': cls_id,
                    'tooth_key': tooth_key,
                    'tooth_short': tooth_short
                })
                
        # Check label consistency anomalies
        for t_short, count in tooth_counts.items():
            if count > 1:
                anomalies.append({
                    'image': img_path.name,
                    'tooth_short': t_short,
                    'count': count,
                    'reason': 'Duplicate tooth position label mapped on single image'
                })
            if count > 5:
                anomalies.append({
                    'image': img_path.name,
                    'tooth_short': t_short,
                    'count': count,
                    'reason': 'More than 5 brackets mapped to same tooth'
                })
                
        # Annotate image
        annotated = img.copy()
        for p in preds:
            x1, y1, x2, y2 = [int(v) for v in p['bbox']]
            is_correct = (p['class'] == 0)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            
            label_text = f"{p['tooth_short']} | {'Correct' if is_correct else 'Incorrect'} | {p['conf']:.2f}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label_text, (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(annotated, label_text, (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        
        rec = {
            'image_name': img_path.name,
            'annotated_img': annotated,
            'preds_count': len(preds),
            'gt_count': len(gt_classes),
            'has_cls1_gt': 1 in gt_classes,
            'has_cls1_pred': any(p['class'] == 1 for p in preds),
            'has_low_conf': any(0.25 <= p['conf'] < 0.45 for p in preds),
            'is_correct_only': (len(preds) > 0 and all(p['class'] == 0 for p in preds))
        }
        
        annotated_records.append(rec)
        
        # Categorize
        if rec['is_correct_only'] and len(cases['correct_only']) < 3:
            cases['correct_only'].append(rec)
        if rec['has_cls1_pred'] and len(cases['has_incorrect']) < 3:
            cases['has_incorrect'].append(rec)
        if rec['preds_count'] >= 6 and len(cases['multi_bracket']) < 3:
            cases['multi_bracket'].append(rec)
        if rec['has_low_conf'] and len(cases['low_confidence']) < 3:
            cases['low_confidence'].append(rec)
        if (rec['has_cls1_gt'] != rec['has_cls1_pred']) and len(cases['failure_case']) < 3:
            cases['failure_case'].append(rec)
            
    # Save candidate images
    saved_candidates = []
    for cat_name, recs in cases.items():
        for i, rec in enumerate(recs):
            out_name = f"{cat_name}_{i+1:02d}_{rec['image_name']}"
            cv2.imwrite(str(OUT_DIR / out_name), rec['annotated_img'])
            saved_candidates.append((cat_name, out_name, rec['annotated_img']))
            
    # Select 12 candidates for contact sheet
    contact_candidates = saved_candidates[:12]
    
    # Build 3x4 contact sheet
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=200)
    fig.suptitle('Qualitative Candidates Contact Sheet (12 Cases)', fontsize=18, fontweight='bold')
    
    for idx, (cat_name, out_name, img_bgr) in enumerate(contact_candidates):
        r, c = idx // 4, idx % 4
        ax = axes[r, c]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f"[{cat_name}]\n{out_name[:25]}", fontsize=10, fontweight='bold')
        ax.axis('off')
        
    for idx in range(len(contact_candidates), 12):
        r, c = idx // 4, idx % 4
        axes[r, c].axis('off')
        
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'contact_sheet.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    with open(OUT_DIR / 'tooth_label_anomalies.json', 'w') as f:
        json.dump(anomalies, f, indent=2)
        
    print(f"Phase 6 complete! Saved candidate overlays and contact sheet in {OUT_DIR}")
    print(f"Logged {len(anomalies)} tooth label anomalies to tooth_label_anomalies.json")

if __name__ == '__main__':
    main()
