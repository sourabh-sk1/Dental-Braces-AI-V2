"""
Dataset Statistics Script (Phase 2)
===================================
Generates results/dataset_stats.json and results/dataset_stats.md
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np

DATASET_DIR = Path('braces_dataset_v3')
RESULTS_DIR = Path('results')

def get_image_info(img_path: Path):
    with Image.open(img_path) as img:
        w, h = img.size
    return w, h, round(w / h, 4)

def count_boxes(lbl_path: Path):
    counts = {0: 0, 1: 0}
    if lbl_path.exists():
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in counts:
                        counts[cls_id] += 1
    return counts

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    
    splits = ['train', 'valid', 'test']
    stats = {}
    all_widths = []
    all_heights = []
    all_aspects = []
    
    # Read manifest for source mapping
    manifest_file = Path('split_manifest.csv')
    sources_per_split = {'train': set(), 'valid': set(), 'test': set()}
    if manifest_file.exists():
        import csv
        with open(manifest_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sources_per_split[row['split']].add(row['source_id'])
                
    for s in splits:
        img_dir = DATASET_DIR / s / 'images'
        lbl_dir = DATASET_DIR / s / 'labels'
        
        img_paths = sorted(list(img_dir.glob('*')))
        c0 = 0
        c1 = 0
        
        widths = []
        heights = []
        aspects = []
        
        for p in img_paths:
            w, h, ar = get_image_info(p)
            widths.append(w)
            heights.append(h)
            aspects.append(ar)
            
            all_widths.append(w)
            all_heights.append(h)
            all_aspects.append(ar)
            
            lbl_p = lbl_dir / f"{p.stem}.txt"
            counts = count_boxes(lbl_p)
            c0 += counts[0]
            c1 += counts[1]
            
        num_imgs = len(img_paths)
        num_sources = len(sources_per_split[s]) if sources_per_split[s] else num_imgs
        aug_factor = round(num_imgs / num_sources, 2) if num_sources > 0 else 1.0
        tot_boxes = c0 + c1
        
        stats[s] = {
            'sources': num_sources,
            'images': num_imgs,
            'augmentation_factor': aug_factor,
            'boxes_class_0_correct': c0,
            'boxes_class_1_incorrect': c1,
            'total_boxes': tot_boxes,
            'boxes_per_image': round(tot_boxes / num_imgs, 2) if num_imgs > 0 else 0,
            'resolution': {
                'min_width': int(np.min(widths)),
                'max_width': int(np.max(widths)),
                'median_width': float(np.median(widths)),
                'min_height': int(np.min(heights)),
                'max_height': int(np.max(heights)),
                'median_height': float(np.median(heights)),
                'median_aspect_ratio': float(np.median(aspects))
            }
        }
        
    overall_stats = {
        'splits': stats,
        'overall_resolution': {
            'min_width': int(np.min(all_widths)),
            'max_width': int(np.max(all_widths)),
            'median_width': float(np.median(all_widths)),
            'min_height': int(np.min(all_heights)),
            'max_height': int(np.max(all_heights)),
            'median_height': float(np.median(all_heights)),
            'min_aspect_ratio': float(np.min(all_aspects)),
            'max_aspect_ratio': float(np.max(all_aspects)),
            'median_aspect_ratio': float(np.median(all_aspects))
        }
    }
    
    with open(RESULTS_DIR / 'dataset_stats.json', 'w') as f:
        json.dump(overall_stats, f, indent=2)
        
    # Write dataset_stats.md
    md_content = f"""# Dataset Statistics (braces_dataset_v3)

## Summary Table

| Split | Sources | Images | Aug. Factor | Class 0 (Correct) | Class 1 (Incorrect) | Total Boxes | Boxes / Image |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Train** | {stats['train']['sources']} | {stats['train']['images']} | {stats['train']['augmentation_factor']}x | {stats['train']['boxes_class_0_correct']} | {stats['train']['boxes_class_1_incorrect']} | {stats['train']['total_boxes']} | {stats['train']['boxes_per_image']} |
| **Validation** | {stats['valid']['sources']} | {stats['valid']['images']} | {stats['valid']['augmentation_factor']}x | {stats['valid']['boxes_class_0_correct']} | {stats['valid']['boxes_class_1_incorrect']} | {stats['valid']['total_boxes']} | {stats['valid']['boxes_per_image']} |
| **Test** | {stats['test']['sources']} | {stats['test']['images']} | {stats['test']['augmentation_factor']}x | {stats['test']['boxes_class_0_correct']} | {stats['test']['boxes_class_1_incorrect']} | {stats['test']['total_boxes']} | {stats['test']['boxes_per_image']} |

## Image Resolution & Aspect Ratio

- **Min Resolution**: {overall_stats['overall_resolution']['min_width']}x{overall_stats['overall_resolution']['min_height']}
- **Median Resolution**: {int(overall_stats['overall_resolution']['median_width'])}x{int(overall_stats['overall_resolution']['median_height'])}
- **Max Resolution**: {overall_stats['overall_resolution']['max_width']}x{overall_stats['overall_resolution']['max_height']}
- **Median Aspect Ratio (W/H)**: {overall_stats['overall_resolution']['median_aspect_ratio']:.4f}
"""
    with open(RESULTS_DIR / 'dataset_stats.md', 'w') as f:
        f.write(md_content)
        
    print("Phase 2 complete! dataset_stats.json and dataset_stats.md written.")

if __name__ == '__main__':
    main()
