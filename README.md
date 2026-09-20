# Dental Braces Detection AI

A research-focused object-detection pipeline for orthodontic bracket detection in dental imagery. This repository implements a leakage-safe dataset workflow, reproducible training and evaluation scripts, baseline comparisons, and a complete ablation study for the final model configuration.

## Overview

This project was built to address a critical data-leakage issue in the original dataset pipeline by enforcing a source-disjoint split before augmentation and model training. The result is a cleaner and more defensible evaluation setup for dental brace detection.

The repository includes:
- Source-based dataset splitting with leakage checks
- Training and validation of the primary YOLOv8 pipeline
- Test-set evaluation and error analysis
- Baseline model comparisons
- A 12-run ablation grid covering class-weight and mosaic variations across seeds
- Evidence files for paper-ready reporting and reproducibility

## Key updates included in this version

- Fixed the dataset leakage problem by splitting on source identity before augmentation
- Kept the original source images and labels untouched in the original dataset folder
- Wrote the transformed leak-safe dataset to [braces_dataset_v3](braces_dataset_v3)
- Restricted augmentation to the training split only
- Added final evaluation and summary reporting in [results](results)
- Completed the full ablation study across the required 2 x 2 x 3 design
- Pinned the environment dependencies and added project licensing files
- Cleaned the project documentation to reflect only evidence-backed outputs

## Methodology

1. Source-aware split: images were partitioned by source ID to prevent near-duplicate examples from appearing across train/val/test.
2. Training-only augmentation: augmentations were applied only to the training split to avoid leakage into validation and testing.
3. Benchmarking: the main model was trained and evaluated on the leak-free split.
4. Comparison: YOLOv8n, YOLOv8s, YOLOv5nu, and a scratch-trained YOLOv8n baseline were compared under the same dataset conditions.
5. Ablation study: class-weight and mosaic variations were tested across multiple seeds to quantify sensitivity of the final design.

## Current evidence-backed results

All values below are taken from the generated outputs in [results](results) and represent the current repository state.

### Main model on the leak-free test split

| Metric | Value |
| :--- | ---: |
| Test images | 77 |
| Test boxes | 422 |
| Precision | 0.8252 |
| Recall | 0.7471 |
| mAP@50 | 0.7511 |
| mAP@50-95 | 0.4835 |

Source: [results/evaluation_report.csv](results/evaluation_report.csv)

### Dataset split summary

| Split | Sources | Images | Boxes |
| :--- | ---: | ---: | ---: |
| Train | 359 | 1,741 | 9,936 |
| Validation | 77 | 77 | 452 |
| Test | 77 | 77 | 422 |

Source: [results/split_summary.json](results/split_summary.json)

### Baseline comparison

| Model | Precision | Recall | mAP@50 | mAP@50-95 | CPU latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| yolov8n (main pretrained) | 0.8252 | 0.7471 | 0.7511 | 0.4835 | 23.68 |
| yolov8s | 0.8394 | 0.8350 | 0.8142 | 0.5373 | 49.59 |
| yolov5nu | 0.8375 | 0.8433 | 0.7970 | 0.5262 | 23.97 |
| yolov8n_scratch | 0.7955 | 0.8454 | 0.8010 | 0.4939 | 21.90 |

Source: [results/baselines.csv](results/baselines.csv)

### Latency benchmark

| Metric | Value |
| :--- | ---: |
| Native device (MPS) mean latency | 12.11 ms |
| Native device median latency | 9.90 ms |
| Forced CPU mean latency | 19.43 ms |
| Forced CPU median latency | 18.96 ms |
| Model weight size | 5.94 MB |
| Parameters | 3,011,238 |
| GFLOPs | 8.19 |

Source: [results/latency.json](results/latency.json)

### Ablation study status

The required ablation matrix has been completed as a full 12-run grid:
- class loss weights: 0.5, 1.2
- mosaic settings: 1.0, 0.0
- seeds: 0, 1, 2

The generated ablation summary is stored in [results/ablation_summary.csv](results/ablation_summary.csv), and the per-run evidence is in [results/ablation_runs.csv](results/ablation_runs.csv).

## Repository structure

- [braces_dataset_fixed_yolov8](braces_dataset_fixed_yolov8): original dataset folder kept intact
- [braces_dataset_v3](braces_dataset_v3): leakage-safe split dataset written for training and evaluation
- [results](results): evidence files for metrics, split summary, latency, error analysis, and ablations
- [runs](runs): training and validation artifacts
- [teeth-braces-ai](teeth-braces-ai): reference project structure and model utilities
- [split_by_source.py](split_by_source.py): source-disjoint dataset splitting pipeline
- [train_main.py](train_main.py): main training script
- [evaluate_main.py](evaluate_main.py): evaluation pipeline
- [train_baselines.py](train_baselines.py): baseline comparison runs
- [run_ablation.py](run_ablation.py): ablation study runner
- [error_analysis.py](error_analysis.py): error breakdown and analysis
- [benchmark_latency.py](benchmark_latency.py): latency benchmarking
- [generate_paper_numbers.py](generate_paper_numbers.py): report-generation helper

## Reproduction

```bash
python split_by_source.py
python dataset_stats.py
python train_main.py
python evaluate_main.py
python error_analysis.py
python qualitative.py
python benchmark_latency.py
python train_baselines.py
python run_ablation.py
python generate_paper_numbers.py
```

## Validation

The repository test suite was validated in the project environment with:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=teeth-braces-ai pytest teeth-braces-ai/tests/ -q
```

This passed successfully in the verified environment.

## Research note and limitations

This project is intended as a research prototype for automated bracket detection in dental imagery. It is not a clinical decision-support system and should not be used as a substitute for professional dental evaluation, diagnosis, or treatment planning.

## Data and licensing notice

The repository includes the original dataset folder [braces_dataset_fixed_yolov8](braces_dataset_fixed_yolov8) and the derived leak-safe dataset [braces_dataset_v3](braces_dataset_v3). Before broader public release, verify any dataset consent, licensing, and provenance conditions associated with the original source material and derived annotations.

This repository includes an MIT license in [LICENSE](LICENSE).

## Project status

The project is in a completed evidence-backed state for the year’s leakage-fix and evaluation workflow, with generated metrics and split artifacts preserved in [results](results).
