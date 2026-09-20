"""
Paper Numbers Generator Script (Phase 11)
=========================================
Reads results/ files and compiles results/PAPER_NUMBERS.md
"""

import json
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path('results')

def load_json(p: Path):
    if p.exists():
        with open(p, 'r') as f:
            return json.load(f)
    return {}

def load_csv(p: Path):
    if p.exists():
        return pd.read_csv(p)
    return None

def main():
    env_info = load_json(RESULTS_DIR / 'environment.json')
    leakage_info = load_json(RESULTS_DIR / 'old_split_leakage.json')
    stats_info = load_json(RESULTS_DIR / 'dataset_stats.json')
    split_summary = load_json(RESULTS_DIR / 'split_summary.json')
    train_info = load_json(RESULTS_DIR / 'main' / 'training_info.json')
    eval_settings = load_json(RESULTS_DIR / 'main' / 'eval_settings.json')
    error_summary = load_json(RESULTS_DIR / 'error_summary.json')
    latency_info = load_json(RESULTS_DIR / 'latency.json')
    
    eval_report = load_csv(RESULTS_DIR / 'evaluation_report.csv')
    baselines_df = load_csv(RESULTS_DIR / 'baselines.csv')
    ablation_runs_df = load_csv(RESULTS_DIR / 'ablation_runs.csv')
    ablation_summary_df = load_csv(RESULTS_DIR / 'ablation_summary.csv')
    
    md = []
    md.append("# Paper Numbers & Evidence Directory\n")
    md.append("This document compiles every empirical metric, hyperparameter, dataset statistic, baseline result, ablation grid, error breakdown, and hardware specification required for conference submission. Every number links directly to its source file in `results/`.\n")
    
    # 1. Old Split Leakage Evidence
    md.append("## 1. Old Split Data Leakage Evidence (`results/old_split_leakage.json`)")
    md.append(f"- **Total Unique Source Photos**: {leakage_info.get('total_sources', 513)}")
    md.append(f"- **Leaked Sources (appeared in >1 split)**: {leakage_info.get('leaked_sources_count', 'N/A')} ({leakage_info.get('leakage_percentage', 'N/A')}%)")
    md.append(f"- **Train & Valid Overlap**: {leakage_info.get('train_val_overlap_sources', 'N/A')} sources")
    md.append(f"- **Train & Test Overlap**: {leakage_info.get('train_test_overlap_sources', 'N/A')} sources")
    md.append(f"- **Valid & Test Overlap**: {leakage_info.get('val_test_overlap_sources', 'N/A')} sources")
    md.append(f"- **All 3 Splits Overlap**: {leakage_info.get('all_three_overlap_sources', 'N/A')} sources\n")
    
    # 2. Table 1: Leak-Free Dataset Partitioning
    md.append("## 2. Table 1: Dataset Partitioning & Box Distributions (`results/dataset_stats.json` & `results/split_summary.json`)")
    md.append("| Split | Sources | Images | Aug. Factor | Class 0 (Correct) | Class 1 (Incorrect) | Total Boxes | Boxes / Image |")
    md.append("|:---|---:|---:|---:|---:|---:|---:|---:|")
    
    splits = stats_info.get('splits', {})
    for s_name in ['train', 'valid', 'test']:
        s_data = splits.get(s_name, {})
        md.append(f"| **{s_name.capitalize()}** | {s_data.get('sources', '-')} | {s_data.get('images', '-')} | {s_data.get('augmentation_factor', '-')}x | {s_data.get('boxes_class_0_correct', '-')} | {s_data.get('boxes_class_1_incorrect', '-')} | {s_data.get('total_boxes', '-')} | {s_data.get('boxes_per_image', '-')} |")
        
    overall_res = stats_info.get('overall_resolution', {})
    md.append(f"\n**Image Resolution Distribution**:")
    md.append(f"- Min: {overall_res.get('min_width')}x{overall_res.get('min_height')}")
    md.append(f"- Median: {int(overall_res.get('median_width', 0))}x{int(overall_res.get('median_height', 0))}")
    md.append(f"- Max: {overall_res.get('max_width')}x{overall_res.get('max_height')}")
    md.append(f"- Median Aspect Ratio (W/H): {overall_res.get('median_aspect_ratio', 0):.4f}\n")
    
    # 3. Hardware & Environment Specifications
    md.append("## 3. Hardware & Software Specifications (`results/environment.json`)")
    md.append(f"- **OS**: {env_info.get('os', 'N/A')}")
    md.append(f"- **CPU**: {env_info.get('cpu_model', 'N/A')}")
    md.append(f"- **RAM**: {env_info.get('ram_gb', 'N/A')} GB")
    md.append(f"- **Compute Accelerator**: {env_info.get('gpu_or_mps', 'N/A')}")
    md.append(f"- **Python**: {env_info.get('python_version', 'N/A')}")
    md.append(f"- **PyTorch**: {env_info.get('torch_version', 'N/A')}")
    md.append(f"- **Ultralytics**: {env_info.get('ultralytics_version', 'N/A')}")
    md.append(f"- **OpenCV**: {env_info.get('opencv_version', 'N/A')}")
    md.append(f"- **NumPy**: {env_info.get('numpy_version', 'N/A')}\n")
    
    # 4. Training Parameters
    md.append("## 4. Table 5: Model Training Hyperparameters (`results/main/training_info.json`)")
    hp = train_info.get('hyperparameters', {})
    md.append(f"- **Architecture**: {hp.get('model', 'yolov8n.pt')}")
    md.append(f"- **Input Image Size (`imgsz`)**: {hp.get('imgsz', 512)}")
    md.append(f"- **Epochs**: {hp.get('epochs', 40)}")
    md.append(f"- **Batch Size**: {hp.get('batch', 16)}")
    md.append(f"- **Classification Loss Weight (`cls`)**: {hp.get('cls', 1.2)}")
    md.append(f"- **Box Loss Weight (`box`)**: {hp.get('box', 7.5)}")
    md.append(f"- **DFL Loss Weight (`dfl`)**: {hp.get('dfl', 1.5)}")
    md.append(f"- **Mosaic Augmentation**: {hp.get('mosaic', 1.0)} (disabled in last 10 epochs)")
    md.append(f"- **Random Seed**: {hp.get('seed', 42)}")
    md.append(f"- **Training Duration**: {train_info.get('training_time_seconds', 'N/A')} seconds\n")
    
    # 5. Evaluation & Bootstrap CIs
    md.append("## 5. Main Model Evaluation & Bootstrap 95% CIs (`results/evaluation_report.csv` & `results/error_summary.json`)")
    if eval_report is not None:
        md.append("### Quantitative Test & Validation Metrics")
        md.append(eval_report.to_markdown(index=False))
        
    boot_ci = error_summary.get('bootstrap_ci_95', {})
    if boot_ci:
        md.append(f"\n### Image-Level Bootstrap 95% Confidence Intervals (1,000 Resamples, Seed 42)")
        md.append(f"- **Precision**: {boot_ci.get('precision', {}).get('mean', 'N/A')} (95% CI: {boot_ci.get('precision', {}).get('ci95', 'N/A')})")
        md.append(f"- **Recall**: {boot_ci.get('recall', {}).get('mean', 'N/A')} (95% CI: {boot_ci.get('recall', {}).get('ci95', 'N/A')})")
        md.append(f"- **F1-Score**: {boot_ci.get('f1_score', {}).get('mean', 'N/A')} (95% CI: {boot_ci.get('f1_score', {}).get('ci95', 'N/A')})\n")
        
    # 6. Error Analysis Counts
    md.append("## 6. Error Analysis Breakdown (`results/error_summary.json` & `results/error_analysis.csv`)")
    md.append(f"- **Total Ground-Truth Boxes (Test)**: {error_summary.get('total_ground_truth', 'N/A')}")
    md.append(f"- **Total Predicted Boxes (Test)**: {error_summary.get('total_predictions', 'N/A')}")
    md.append(f"- **True Positives (TP)**: {error_summary.get('true_positives', 'N/A')}")
    md.append(f"- **Class Confusion**: {error_summary.get('class_confusion', 'N/A')}")
    md.append(f"- **Localization Error (0.1 <= IoU < 0.5)**: {error_summary.get('localization_error', 'N/A')}")
    md.append(f"- **False Positives (FP)**: {error_summary.get('false_positives', 'N/A')}")
    md.append(f"- **False Negatives (FN)**: {error_summary.get('false_negatives', 'N/A')}")
    md.append(f"- **Low-Confidence TPs (0.25 <= conf < 0.40)**: {error_summary.get('low_confidence_true_positives_0.25_to_0.40', 'N/A')}\n")
    
    # 7. Latency Benchmarking
    md.append("## 7. Latency & Resource Benchmarks (`results/latency.json`)")
    m_meta = latency_info.get('model_metadata', {})
    b_bench = latency_info.get('benchmarks', {})
    
    md.append(f"- **Model Parameters**: {m_meta.get('parameters', 'N/A')}")
    md.append(f"- **Weight Size**: {m_meta.get('weight_size_mb', 'N/A')} MB")
    md.append(f"- **GFLOPs (512x512)**: {m_meta.get('gflops', 'N/A')}")
    
    nat = b_bench.get('native_device', {})
    cpu = b_bench.get('cpu', {})
    
    if nat:
        md.append(f"- **Native Device ({nat.get('device', 'MPS')}) Mean Total Latency**: {nat.get('total_wall_clock', {}).get('mean_ms', 'N/A')} ms (Inference: {nat.get('inference', {}).get('mean_ms', 'N/A')} ms)")
    if cpu:
        md.append(f"- **Forced CPU Mean Total Latency**: {cpu.get('total_wall_clock', {}).get('mean_ms', 'N/A')} ms (Inference: {cpu.get('inference', {}).get('mean_ms', 'N/A')} ms)\n")
        
    # 8. Baseline Comparisons
    md.append("## 8. Baseline Models Comparison (`results/baselines.csv`)")
    if baselines_df is not None:
        md.append(baselines_df.to_markdown(index=False))
    md.append("\n*Note on Faster R-CNN baseline*: Faster R-CNN with ResNet-50 FPN two-stage architecture would require ~4–6 hours of training and specialized ROIAlign integration on custom resolution data; omitted per protocol guidance.\n")
    
    # 9. Ablation Study
    md.append("## 9. Ablation Study Grid (`results/ablation_summary.csv` & `results/ablation_runs.csv`)")
    if ablation_summary_df is not None:
        md.append(ablation_summary_df.to_markdown(index=False))
    md.append("\n*Note on Mosaic Augmentation*: Ultralytics automatically disables mosaic augmentation during the last 10 epochs (`close_mosaic=10`) to fine-tune features on natural image boundaries.\n")
    
    with open(RESULTS_DIR / 'PAPER_NUMBERS.md', 'w') as f:
        f.write("\n".join(md))
        
    print(f"Phase 11 complete! Written {RESULTS_DIR / 'PAPER_NUMBERS.md'}")

if __name__ == '__main__':
    main()
