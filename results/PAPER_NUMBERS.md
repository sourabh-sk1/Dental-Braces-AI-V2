# Paper Numbers & Evidence Directory

This document compiles every empirical metric, hyperparameter, dataset statistic, baseline result, ablation grid, error breakdown, and hardware specification required for conference submission. Every number links directly to its source file in `results/`.

## 1. Old Split Data Leakage Evidence (`results/old_split_leakage.json`)
- **Total Unique Source Photos**: 513
- **Leaked Sources (appeared in >1 split)**: 421 (82.07%)
- **Train & Valid Overlap**: 340 sources
- **Train & Test Overlap**: 203 sources
- **Valid & Test Overlap**: 122 sources
- **All 3 Splits Overlap**: 122 sources

## 2. Table 1: Dataset Partitioning & Box Distributions (`results/dataset_stats.json` & `results/split_summary.json`)
| Split | Sources | Images | Aug. Factor | Class 0 (Correct) | Class 1 (Incorrect) | Total Boxes | Boxes / Image |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 359 | 1741 | 4.85x | 6738 | 3198 | 9936 | 5.71 |
| **Valid** | 77 | 77 | 1.0x | 299 | 153 | 452 | 5.87 |
| **Test** | 77 | 77 | 1.0x | 275 | 147 | 422 | 5.48 |

**Image Resolution Distribution**:
- Min: 512x512
- Median: 512x512
- Max: 512x512
- Median Aspect Ratio (W/H): 1.0000

## 3. Hardware & Software Specifications (`results/environment.json`)
- **OS**: Darwin 27.0.0 (arm64)
- **CPU**: Apple M4
- **RAM**: 16.0 GB
- **Compute Accelerator**: Apple Silicon MPS
- **Python**: 3.13.9
- **PyTorch**: 2.14.0
- **Ultralytics**: 8.4.150
- **OpenCV**: 5.0.0
- **NumPy**: 2.5.3

## 4. Table 5: Model Training Hyperparameters (`results/main/training_info.json`)
- **Architecture**: yolov8n.pt
- **Input Image Size (`imgsz`)**: 512
- **Epochs**: 40
- **Batch Size**: 16
- **Classification Loss Weight (`cls`)**: 1.2
- **Box Loss Weight (`box`)**: 7.5
- **DFL Loss Weight (`dfl`)**: 1.5
- **Mosaic Augmentation**: 1.0 (disabled in last 10 epochs)
- **Random Seed**: 42
- **Training Duration**: 1370.61 seconds

## 5. Main Model Evaluation & Bootstrap 95% CIs (`results/evaluation_report.csv` & `results/error_summary.json`)
### Quantitative Test & Validation Metrics
| split   | class           |   images |   instances |   precision |   recall |   map50 |   map50_95 |
|:--------|:----------------|---------:|------------:|------------:|---------:|--------:|-----------:|
| test    | Correct Brace   |       77 |         275 |      0.8156 |   0.6914 |  0.7326 |     0.465  |
| test    | Incorrect Brace |       77 |         147 |      0.8348 |   0.8027 |  0.7695 |     0.5019 |
| test    | all             |       77 |         422 |      0.8252 |   0.7471 |  0.7511 |     0.4835 |
| valid   | Correct Brace   |       77 |         299 |      0.8595 |   0.8796 |  0.8667 |     0.5637 |
| valid   | Incorrect Brace |       77 |         153 |      0.8841 |   0.902  |  0.9223 |     0.6442 |
| valid   | all             |       77 |         452 |      0.8718 |   0.8908 |  0.8945 |     0.6039 |

### Image-Level Bootstrap 95% Confidence Intervals (1,000 Resamples, Seed 42)
- **Precision**: 0.7712 (95% CI: [0.714, 0.829])
- **Recall**: 0.8135 (95% CI: [0.7455, 0.8732])
- **F1-Score**: 0.7913 (95% CI: [0.7378, 0.8386])

## 6. Error Analysis Breakdown (`results/error_summary.json` & `results/error_analysis.csv`)
- **Total Ground-Truth Boxes (Test)**: 422
- **Total Predicted Boxes (Test)**: 445
- **True Positives (TP)**: 333
- **Class Confusion**: 12
- **Localization Error (0.1 <= IoU < 0.5)**: 1
- **False Positives (FP)**: 99
- **False Negatives (FN)**: 76
- **Low-Confidence TPs (0.25 <= conf < 0.40)**: 30

## 7. Latency & Resource Benchmarks (`results/latency.json`)
- **Model Parameters**: 3011238
- **Weight Size**: 5.94 MB
- **GFLOPs (512x512)**: 8.19
- **Native Device (mps) Mean Total Latency**: 12.105 ms (Inference: 2.521 ms)
- **Forced CPU Mean Total Latency**: 19.429 ms (Inference: 18.118 ms)

## 8. Baseline Models Comparison (`results/baselines.csv`)
| model                     |   precision |   recall |   map50 |   map50_95 |   parameters |   weight_size_mb |   cpu_latency_ms |
|:--------------------------|------------:|---------:|--------:|-----------:|-------------:|-----------------:|-----------------:|
| yolov8n (main pretrained) |      0.8252 |   0.7471 |  0.7511 |     0.4835 |      3011238 |             5.94 |            23.68 |
| yolov8s                   |      0.8394 |   0.835  |  0.8142 |     0.5373 |     11136374 |            21.46 |            49.59 |
| yolov5nu                  |      0.8375 |   0.8433 |  0.797  |     0.5262 |      2508854 |             5.01 |            23.97 |
| yolov8n_scratch           |      0.7955 |   0.8454 |  0.801  |     0.4939 |      3011238 |             5.94 |            21.9  |

*Note on Faster R-CNN baseline*: Faster R-CNN with ResNet-50 FPN two-stage architecture would require ~4–6 hours of training and specialized ROIAlign integration on custom resolution data; omitted per protocol guidance.

## 9. Ablation Study Grid (`results/ablation_summary.csv` & `results/ablation_runs.csv`)
|   cls |   mosaic |   close_mosaic_epochs |   val_map50_mean |   val_map50_sd |   val_map50_95_mean |   val_map50_95_sd |   test_map50_mean |   test_map50_sd |   test_map50_95_mean |   test_map50_95_sd |
|------:|---------:|----------------------:|-----------------:|---------------:|--------------------:|------------------:|------------------:|----------------:|---------------------:|-------------------:|
|   0.5 |        0 |                    10 |           0.3553 |            nan |              0.2002 |               nan |            0.3403 |             nan |               0.183  |                nan |
|   0.5 |        1 |                    10 |           0.9554 |            nan |              0.6557 |               nan |            0.8369 |             nan |               0.5544 |                nan |
|   1.2 |        0 |                    10 |           0.9428 |            nan |              0.6348 |               nan |            0.8191 |             nan |               0.5281 |                nan |
|   1.2 |        1 |                    10 |           0.9428 |            nan |              0.6348 |               nan |            0.8191 |             nan |               0.5281 |                nan |

*Note on Mosaic Augmentation*: Ultralytics automatically disables mosaic augmentation during the last 10 epochs (`close_mosaic=10`) to fine-tune features on natural image boundaries.
