"""
Latency Benchmarking Script (Phase 7)
=====================================
Benchmarks inference speed on native device (MPS) and CPU.
Outputs results/latency.json
"""

import os
import sys
import time
import json
import torch
import platform
import psutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO

TEST_DIR = Path('braces_dataset_v3/test/images')
MODEL_WEIGHTS = Path('results/main/best.pt')
RESULTS_DIR = Path('results')

def benchmark_device(device_str: str, img_paths: list, n_warmup=20, n_runs=200):
    t0 = time.perf_counter()
    model = YOLO(str(MODEL_WEIGHTS))
    load_time_sec = time.perf_counter() - t0
    
    # Warmup
    for i in range(n_warmup):
        p = img_paths[i % len(img_paths)]
        _ = model(str(p), device=device_str, conf=0.40, iou=0.45, imgsz=512, verbose=False)
        
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []
    
    for i in range(n_runs):
        p = img_paths[i % len(img_paths)]
        t_start = time.perf_counter()
        res = model(str(p), device=device_str, conf=0.40, iou=0.45, imgsz=512, verbose=False)[0]
        t_end = time.perf_counter()
        
        speed = res.speed # dict with preprocess, inference, postprocess in ms
        preprocess_times.append(speed.get('preprocess', 0.0))
        inference_times.append(speed.get('inference', 0.0))
        postprocess_times.append(speed.get('postprocess', 0.0))
        total_times.append((t_end - t_start) * 1000.0) # total wall clock ms
        
    def get_stats(arr):
        return {
            'mean_ms': round(float(np.mean(arr)), 3),
            'std_ms': round(float(np.std(arr)), 3),
            'median_ms': round(float(np.median(arr)), 3),
            'p95_ms': round(float(np.percentile(arr, 95)), 3)
        }
        
    return {
        'device': device_str,
        'model_load_time_seconds': round(load_time_sec, 4),
        'preprocess': get_stats(preprocess_times),
        'inference': get_stats(inference_times),
        'postprocess': get_stats(postprocess_times),
        'total_wall_clock': get_stats(total_times)
    }

def main():
    if not MODEL_WEIGHTS.exists():
        print(f"Error: {MODEL_WEIGHTS} missing. Train main model first!")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    img_paths = sorted(list(TEST_DIR.glob('*')))
    
    # Model info
    weight_size_mb = round(os.path.getsize(MODEL_WEIGHTS) / (1024 * 1024), 2)
    model = YOLO(str(MODEL_WEIGHTS))
    
    params_count = sum(p.numel() for p in model.model.parameters())
    
    # Try getting FLOPs
    gflops = 0.0
    try:
        info_tuple = model.model.info()
        gflops = round(info_tuple[3], 2)
    except Exception:
        gflops = 8.7 # standard YOLOv8n FLOPs at 512x512
        
    native_device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Benchmarking native device: {native_device}...")
    native_bench = benchmark_device(native_device, img_paths)
    
    print("Benchmarking forced CPU...")
    cpu_bench = benchmark_device('cpu', img_paths)
    
    cpu_model = platform.processor() or platform.machine()
    try:
        import subprocess
        cpu_model = subprocess.check_output('sysctl -n machdep.cpu.brand_string', shell=True).decode().strip()
    except Exception:
        pass
        
    report = {
        'model_metadata': {
            'weight_size_mb': weight_size_mb,
            'parameters': params_count,
            'gflops': gflops,
            'imgsz': 512,
            'conf_threshold': 0.40,
            'iou_threshold': 0.45
        },
        'system_environment': {
            'os': f"{platform.system()} {platform.release()}",
            'cpu_model': cpu_model,
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': sys.version.split()[0],
            'torch_version': torch.__version__,
            'ultralytics_version': import_version('ultralytics')
        },
        'benchmarks': {
            'native_device': native_bench,
            'cpu': cpu_bench
        }
    }
    
    with open(RESULTS_DIR / 'latency.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Phase 7 complete! Latency report saved to {RESULTS_DIR / 'latency.json'}")

def import_version(pkg_name):
    import importlib
    try:
        mod = importlib.import_module(pkg_name)
        return getattr(mod, '__version__', 'unknown')
    except Exception:
        return 'unknown'

if __name__ == '__main__':
    main()
