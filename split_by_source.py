"""
Leak-Free Source-Level Split Script (Phase 1)
==============================================
Splits the 513 unique source photos into train (70%), validation (15%), and test (15%)
without data leakage across augmented variants.
"""

import os
import re
import json
import csv
import shutil
import random
import numpy as np
from pathlib import Path
from PIL import Image
import imagehash

# Configuration
OLD_DATASET_DIR = Path('braces_dataset_fixed_yolov8')
NEW_DATASET_DIR = Path('braces_dataset_v3')
RESULTS_DIR = Path('results')
SEED = 42

def extract_source_id(filename: str) -> str:
    """Strip Roboflow suffix .rf.<32-char-hash> from filename."""
    return re.sub(r'\.rf\.[0-9a-fA-F]{32}', '', filename)

def get_class_counts(label_path: Path) -> dict:
    counts = {0: 0, 1: 0}
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in counts:
                        counts[cls_id] += 1
    return counts

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # 1. Collect all images and labels from old dataset
    old_splits = ['train', 'valid', 'test']
    source_map = {}  # source_id -> list of dicts: {'path': Path, 'old_split': str, 'label_path': Path}
    
    for s in old_splits:
        img_dir = OLD_DATASET_DIR / s / 'images'
        lbl_dir = OLD_DATASET_DIR / s / 'labels'
        
        for img_path in sorted(img_dir.glob('*')):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            sid = extract_source_id(img_path.name)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            if sid not in source_map:
                source_map[sid] = []
            
            source_map[sid].append({
                'path': img_path,
                'old_split': s,
                'label_path': lbl_path
            })
            
    print(f"Total unique sources identified: {len(source_map)}")
    assert len(source_map) == 513, f"Expected 513 unique sources, got {len(source_map)}"
    
    # 2. Measure old split leakage
    old_leakage_sources = []
    split_presence = {} # sid -> set of old_splits
    
    for sid, variants in source_map.items():
        splits_found = set(v['old_split'] for v in variants)
        split_presence[sid] = splits_found
        if len(splits_found) > 1:
            old_leakage_sources.append(sid)
            
    train_val_overlap = sum(1 for sid, s_set in split_presence.items() if 'train' in s_set and 'valid' in s_set)
    train_test_overlap = sum(1 for sid, s_set in split_presence.items() if 'train' in s_set and 'test' in s_set)
    val_test_overlap = sum(1 for sid, s_set in split_presence.items() if 'valid' in s_set and 'test' in s_set)
    all_three_overlap = sum(1 for sid, s_set in split_presence.items() if len(s_set) == 3)
    
    old_leakage_report = {
        'total_sources': len(source_map),
        'leaked_sources_count': len(old_leakage_sources),
        'leakage_percentage': round(len(old_leakage_sources) / len(source_map) * 100, 2),
        'train_val_overlap_sources': train_val_overlap,
        'train_test_overlap_sources': train_test_overlap,
        'val_test_overlap_sources': val_test_overlap,
        'all_three_overlap_sources': all_three_overlap,
        'leaked_source_ids': sorted(old_leakage_sources)
    }
    
    with open(RESULTS_DIR / 'old_split_leakage.json', 'w') as f:
        json.dump(old_leakage_report, f, indent=2)
        
    print(f"Old split leakage: {len(old_leakage_sources)} / {len(source_map)} sources ({old_leakage_report['leakage_percentage']}%) appear in >1 split!")
    
    # 3. Stratify sources by presence of Class 1 (Incorrect Brace)
    source_strata = {}
    for sid, variants in source_map.items():
        has_cls1 = False
        for v in variants:
            counts = get_class_counts(v['label_path'])
            if counts[1] > 0:
                has_cls1 = True
                break
        source_strata[sid] = has_cls1
        
    cls1_sources = sorted([sid for sid, has_c1 in source_strata.items() if has_c1])
    cls0_sources = sorted([sid for sid, has_c1 in source_strata.items() if not has_c1])
    
    print(f"Stratification: {len(cls1_sources)} sources with Incorrect Brace (Class 1), {len(cls0_sources)} with only Correct Brace (Class 0)")
    
    # Search seeds 42 to 46 for optimal class balance in val/test
    selected_seed = SEED
    best_split_assignment = None
    best_summary = None
    
    for candidate_seed in range(42, 47):
        rng = random.Random(candidate_seed)
        
        # Shuffle each stratum deterministically
        cls1_shuffled = cls1_sources.copy()
        cls0_shuffled = cls0_sources.copy()
        rng.shuffle(cls1_shuffled)
        rng.shuffle(cls0_shuffled)
        
        def split_list(lst, r_train=0.70, r_val=0.15):
            n = len(lst)
            n_train = int(round(n * r_train))
            n_val = int(round(n * r_val))
            return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]
            
        c1_tr, c1_va, c1_te = split_list(cls1_shuffled)
        c0_tr, c0_va, c0_te = split_list(cls0_shuffled)
        
        train_sids = set(c1_tr + c0_tr)
        val_sids = set(c1_va + c0_va)
        test_sids = set(c1_te + c0_te)
        
        # Compute box counts for val and test
        def compute_split_stats(sids, is_eval_split=False):
            img_count = 0
            box_cls0 = 0
            box_cls1 = 0
            for sid in sids:
                variants = sorted(source_map[sid], key=lambda x: x['path'].name)
                selected_variants = [variants[0]] if is_eval_split else variants
                img_count += len(selected_variants)
                for v in selected_variants:
                    counts = get_class_counts(v['label_path'])
                    box_cls0 += counts[0]
                    box_cls1 += counts[1]
            return {
                'sources': len(sids),
                'images': img_count,
                'boxes_cls0': box_cls0,
                'boxes_cls1': box_cls1,
                'total_boxes': box_cls0 + box_cls1,
                'boxes_per_image': round((box_cls0 + box_cls1) / img_count, 2) if img_count > 0 else 0
            }
            
        tr_stats = compute_split_stats(train_sids, is_eval_split=False)
        va_stats = compute_split_stats(val_sids, is_eval_split=True)
        te_stats = compute_split_stats(test_sids, is_eval_split=True)
        
        min_boxes = min(va_stats['boxes_cls0'], va_stats['boxes_cls1'], te_stats['boxes_cls0'], te_stats['boxes_cls1'])
        
        print(f"Seed {candidate_seed}: Val boxes (cls0={va_stats['boxes_cls0']}, cls1={va_stats['boxes_cls1']}), Test boxes (cls0={te_stats['boxes_cls0']}, cls1={te_stats['boxes_cls1']})")
        
        if min_boxes >= 50:
            selected_seed = candidate_seed
            best_split_assignment = {'train': train_sids, 'valid': val_sids, 'test': test_sids}
            best_summary = {'train': tr_stats, 'valid': va_stats, 'test': te_stats}
            break
            
    if best_split_assignment is None:
        selected_seed = 42
        # fallback to 42
        rng = random.Random(42)
        cls1_shuffled = cls1_sources.copy()
        cls0_shuffled = cls0_sources.copy()
        rng.shuffle(cls1_shuffled)
        rng.shuffle(cls0_shuffled)
        c1_tr, c1_va, c1_te = split_list(cls1_shuffled)
        c0_tr, c0_va, c0_te = split_list(cls0_shuffled)
        best_split_assignment = {
            'train': set(c1_tr + c0_tr),
            'valid': set(c1_va + c0_va),
            'test': set(c1_te + c0_te)
        }
        best_summary = {
            'train': compute_split_stats(best_split_assignment['train'], False),
            'valid': compute_split_stats(best_split_assignment['valid'], True),
            'test': compute_split_stats(best_split_assignment['test'], True)
        }
        
    print(f"\nSelected Seed: {selected_seed}")
    print(f"Split sources count - Train: {len(best_split_assignment['train'])}, Valid: {len(best_split_assignment['valid'])}, Test: {len(best_split_assignment['test'])}")
    
    # 4. Write braces_dataset_v3 and split_manifest.csv
    if NEW_DATASET_DIR.exists():
        shutil.rmtree(NEW_DATASET_DIR)
        
    manifest_rows = []
    
    for split_name, sids in best_split_assignment.items():
        img_out_dir = NEW_DATASET_DIR / split_name / 'images'
        lbl_out_dir = NEW_DATASET_DIR / split_name / 'labels'
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)
        
        is_eval = (split_name in ['valid', 'test'])
        
        for sid in sorted(sids):
            variants = sorted(source_map[sid], key=lambda x: x['path'].name)
            selected_variants = [variants[0]] if is_eval else variants
            
            for idx, v in enumerate(selected_variants):
                target_img = img_out_dir / v['path'].name
                target_lbl = lbl_out_dir / f"{v['path'].stem}.txt"
                
                shutil.copy2(v['path'], target_img)
                if v['label_path'].exists():
                    shutil.copy2(v['label_path'], target_lbl)
                else:
                    target_lbl.touch()
                    
                manifest_rows.append({
                    'source_id': sid,
                    'split': split_name,
                    'file': v['path'].name,
                    'original_split': v['old_split'],
                    'is_eval_sample': is_eval
                })
                
    # Save split_manifest.csv
    with open('split_manifest.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_id', 'split', 'file', 'original_split', 'is_eval_sample'])
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    # Write data.yaml
    data_yaml_content = f"""path: .
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: Correct Brace
  1: Incorrect Brace
"""
    with open(NEW_DATASET_DIR / 'data.yaml', 'w') as f:
        f.write(data_yaml_content)
        
    # Write split_summary.json
    with open(RESULTS_DIR / 'split_summary.json', 'w') as f:
        json.dump({
            'seed_used': selected_seed,
            'summary': best_summary
        }, f, indent=2)
        
    print("\nDataset braces_dataset_v3 created successfully!")
    print("Writing split_summary.json:")
    print(json.dumps(best_summary, indent=2))
    
    # 5. Perceptual-hash near-duplicate verification across splits
    print("\nRunning perceptual-hash cross-split near-duplicate check...")
    hashes = {} # split -> list of (file, hash)
    
    for split_name in ['train', 'valid', 'test']:
        img_dir = NEW_DATASET_DIR / split_name / 'images'
        hashes[split_name] = []
        for img_p in sorted(img_dir.glob('*')):
            try:
                with Image.open(img_p) as img:
                    h = imagehash.phash(img)
                    hashes[split_name].append((img_p.name, h, extract_source_id(img_p.name)))
            except Exception as e:
                pass
                
    # Compare pairs between splits
    split_pairs = [('train', 'valid'), ('train', 'test'), ('valid', 'test')]
    near_dupes = []
    
    for s1, s2 in split_pairs:
        for f1, h1, sid1 in hashes[s1]:
            for f2, h2, sid2 in hashes[s2]:
                dist = h1 - h2
                if dist <= 5: # Small Hamming distance threshold
                    near_dupes.append({
                        'split1': s1, 'file1': f1, 'source1': sid1,
                        'split2': s2, 'file2': f2, 'source2': sid2,
                        'hamming_distance': int(dist)
                    })
                    
    print(f"Perceptual hash check complete. Found {len(near_dupes)} pairs with Hamming distance <= 5 across splits.")
    if near_dupes:
        print("Sample near-duplicate cross-split pairs:")
        for nd in near_dupes[:5]:
            print(f"  [{nd['split1']} vs {nd['split2']}] d={nd['hamming_distance']}: {nd['file1']} ({nd['source1']}) <-> {nd['file2']} ({nd['source2']})")
            
    with open(RESULTS_DIR / 'phash_cross_split_check.json', 'w') as f:
        json.dump(near_dupes, f, indent=2)

if __name__ == '__main__':
    main()
