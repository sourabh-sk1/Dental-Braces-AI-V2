"""
Error Analysis & Bootstrap CIs Script (Phase 5)
===============================================
Runs on the TEST split at conf=0.25, iou=0.5.
Matches predictions to ground truth with strict partition assertions:
  TP + Class Confusion + Localization Error + FN = GT Count
  TP + Class Confusion + Localization Error + FP = Pred Count

Outputs:
- results/error_analysis.csv
- results/error_summary.json
- results/errors/ (up to 20 visual overlays)
- Bootstrap 95% CIs (1000 resamples, seed 42)
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

DATA_DIR = Path('braces_dataset_v3/test')
MODEL_WEIGHTS = Path('results/main/best.pt')
RESULTS_DIR = Path('results')
ERRORS_DIR = RESULTS_DIR / 'errors'

def xywh2xyxy(bbox, img_w, img_h):
    xc, yc, w, h = bbox
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def main():
    if not MODEL_WEIGHTS.exists():
        print(f"Error: {MODEL_WEIGHTS} missing. Train main model first!")
        return

    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_WEIGHTS))
    
    img_dir = DATA_DIR / 'images'
    lbl_dir = DATA_DIR / 'labels'
    img_paths = sorted(list(img_dir.glob('*')))
    
    total_gt = 0
    total_preds = 0
    
    tp_count = 0
    class_confusion_count = 0
    loc_error_count = 0
    fp_count = 0
    fn_count = 0
    low_conf_tp_count = 0 # conf between 0.25 and 0.40 (not errors, but logged)
    
    error_rows = []
    image_metrics = [] # for bootstrap
    
    overlay_count = 0
    
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        xyxy = xywh2xyxy(bbox, img_w, img_h)
                        gt_boxes.append({'class': cls_id, 'bbox': xyxy, 'matched': False})
                        
        # Model inference
        results = model(str(img_path), conf=0.25, iou=0.50, imgsz=512, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())
                pred_boxes.append({'class': cls_id, 'bbox': xyxy, 'conf': conf, 'matched': False})
                
        total_gt += len(gt_boxes)
        total_preds += len(pred_boxes)
        
        # Candidate matches between pred and gt
        possible_matches = []
        for p_idx, p in enumerate(pred_boxes):
            for g_idx, g in enumerate(gt_boxes):
                iou = compute_iou(p['bbox'], g['bbox'])
                if iou >= 0.1:
                    possible_matches.append((iou, p_idx, g_idx))
                    
        # Sort matches by IoU descending
        possible_matches.sort(key=lambda x: x[0], reverse=True)
        
        img_tp = 0
        img_cls_conf = 0
        img_loc_err = 0
        img_fp = 0
        img_fn = 0
        
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        for iou, p_idx, g_idx in possible_matches:
            if p_idx in matched_pred_indices or g_idx in matched_gt_indices:
                continue
                
            p = pred_boxes[p_idx]
            g = gt_boxes[g_idx]
            
            matched_pred_indices.add(p_idx)
            matched_gt_indices.add(g_idx)
            
            if iou >= 0.5 and p['class'] == g['class']:
                tp_count += 1
                img_tp += 1
                if 0.25 <= p['conf'] < 0.40:
                    low_conf_tp_count += 1
            elif iou >= 0.5 and p['class'] != g['class']:
                class_confusion_count += 1
                img_cls_conf += 1
                error_rows.append({
                    'image': img_path.name,
                    'class': p['class'],
                    'gt_class': g['class'],
                    'box': [round(x, 1) for x in p['bbox']],
                    'conf': round(p['conf'], 4),
                    'iou': round(iou, 4),
                    'error_type': 'class_confusion'
                })
            elif 0.1 <= iou < 0.5 and p['class'] == g['class']:
                loc_error_count += 1
                img_loc_err += 1
                error_rows.append({
                    'image': img_path.name,
                    'class': p['class'],
                    'gt_class': g['class'],
                    'box': [round(x, 1) for x in p['bbox']],
                    'conf': round(p['conf'], 4),
                    'iou': round(iou, 4),
                    'error_type': 'localization_error'
                })
            else:
                # Treated as FP and FN
                fp_count += 1
                fn_count += 1
                img_fp += 1
                img_fn += 1
                
        # Unmatched predictions -> FP
        for p_idx, p in enumerate(pred_boxes):
            if p_idx not in matched_pred_indices:
                fp_count += 1
                img_fp += 1
                # Find max iou with any gt
                max_iou = max([compute_iou(p['bbox'], g['bbox']) for g in gt_boxes], default=0.0)
                error_rows.append({
                    'image': img_path.name,
                    'class': p['class'],
                    'gt_class': -1,
                    'box': [round(x, 1) for x in p['bbox']],
                    'conf': round(p['conf'], 4),
                    'iou': round(max_iou, 4),
                    'error_type': 'false_positive'
                })
                
        # Unmatched ground truths -> FN
        for g_idx, g in enumerate(gt_boxes):
            if g_idx not in matched_gt_indices:
                fn_count += 1
                img_fn += 1
                max_iou = max([compute_iou(g['bbox'], p['bbox']) for p in pred_boxes], default=0.0)
                error_rows.append({
                    'image': img_path.name,
                    'class': g['class'],
                    'gt_class': g['class'],
                    'box': [round(x, 1) for x in g['bbox']],
                    'conf': 0.0,
                    'iou': round(max_iou, 4),
                    'error_type': 'false_negative'
                })
                
        image_metrics.append({
            'tp': img_tp,
            'cls_conf': img_cls_conf,
            'loc_err': img_loc_err,
            'fp': img_fp,
            'fn': img_fn
        })
        
        # Save overlay for up to 20 error cases
        if (img_cls_conf > 0 or img_loc_err > 0 or img_fp > 0 or img_fn > 0) and overlay_count < 20:
            overlay_count += 1
            overlay_img = img.copy()
            
            # Draw GT in Blue
            for g in gt_boxes:
                x1, y1, x2, y2 = [int(v) for v in g['bbox']]
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
            # Draw Pred in Red (Incorrect) / Green (Correct)
            for p in pred_boxes:
                x1, y1, x2, y2 = [int(v) for v in p['bbox']]
                color = (0, 255, 0) if p['class'] == 0 else (0, 0, 255)
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay_img, f"{p['class']}:{p['conf']:.2f}", (x1, max(y1-5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
            cv2.imwrite(str(ERRORS_DIR / f"error_{overlay_count:02d}_{img_path.name}"), overlay_img)
            
    # Mathematical assertions
    gt_sum = tp_count + class_confusion_count + loc_error_count + fn_count
    pred_sum = tp_count + class_confusion_count + loc_error_count + fp_count
    
    assert gt_sum == total_gt, f"GT assertion failed: {gt_sum} != {total_gt}"
    assert pred_sum == total_preds, f"Pred assertion failed: {pred_sum} != {total_preds}"
    
    print(f"Assertions PASSED!")
    print(f"Total GT: {total_gt}, Total Preds: {total_preds}")
    print(f"TP: {tp_count}, Class Confusion: {class_confusion_count}, Loc Error: {loc_error_count}, FP: {fp_count}, FN: {fn_count}")
    print(f"Low-Confidence TPs (conf 0.25-0.40): {low_conf_tp_count}")
    
    # Save results/error_analysis.csv
    df_err = pd.DataFrame(error_rows)
    df_err.to_csv(RESULTS_DIR / 'error_analysis.csv', index=False)
    
    # Compute image-level bootstrap 95% CIs (1000 resamples, seed 42)
    rng = np.random.RandomState(42)
    n_images = len(image_metrics)
    
    boot_p = []
    boot_r = []
    boot_f1 = []
    
    for _ in range(1000):
        boot_idx = rng.choice(n_images, size=n_images, replace=True)
        b_tp = sum(image_metrics[i]['tp'] for i in boot_idx)
        b_fp = sum(image_metrics[i]['fp'] for i in boot_idx)
        b_fn = sum(image_metrics[i]['fn'] for i in boot_idx)
        
        p = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        r = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        boot_p.append(p)
        boot_r.append(r)
        boot_f1.append(f1)
        
    p_ci = [round(np.percentile(boot_p, 2.5), 4), round(np.percentile(boot_p, 97.5), 4)]
    r_ci = [round(np.percentile(boot_r, 2.5), 4), round(np.percentile(boot_r, 97.5), 4)]
    f1_ci = [round(np.percentile(boot_f1, 2.5), 4), round(np.percentile(boot_f1, 97.5), 4)]
    
    summary = {
        'total_ground_truth': total_gt,
        'total_predictions': total_preds,
        'true_positives': tp_count,
        'class_confusion': class_confusion_count,
        'localization_error': loc_error_count,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'low_confidence_true_positives_0.25_to_0.40': low_conf_tp_count,
        'bootstrap_ci_95': {
            'precision': {'mean': round(np.mean(boot_p), 4), 'ci95': p_ci},
            'recall': {'mean': round(np.mean(boot_r), 4), 'ci95': r_ci},
            'f1_score': {'mean': round(np.mean(boot_f1), 4), 'ci95': f1_ci}
        }
    }
    
    with open(RESULTS_DIR / 'error_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Error summary and bootstrap CIs saved to results/error_summary.json")

if __name__ == '__main__':
    main()
