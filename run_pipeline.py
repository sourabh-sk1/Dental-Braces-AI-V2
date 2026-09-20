"""
Master Pipeline Orchestrator (Phases 4 - 11)
============================================
Runs evaluation, error analysis, qualitative visualization,
latency benchmarking, baselines, ablation studies, and paper numbers generation.
"""

import subprocess
import sys
from pathlib import Path

def run_step(cmd, step_name):
    print(f"\n" + "=" * 60)
    print(f"RUNNING STEP: {step_name}")
    print("=" * 60)
    res = subprocess.run([sys.executable] + cmd, check=True)
    print(f"STEP {step_name} COMPLETED WITH EXIT CODE {res.returncode}")

def main():
    # Phase 4: Evaluate Main Model
    run_step(['evaluate_main.py'], 'Phase 4: Evaluate Main Model')
    
    # Phase 5: Error Analysis
    run_step(['error_analysis.py'], 'Phase 5: Error Analysis & Bootstrap CIs')
    
    # Phase 6: Qualitative Visualization
    run_step(['qualitative.py'], 'Phase 6: Qualitative Examples & Contact Sheet')
    
    # Phase 7: Latency Benchmark
    run_step(['benchmark_latency.py'], 'Phase 7: Latency Benchmarking')
    
    # Phase 8: Baselines
    run_step(['train_baselines.py'], 'Phase 8: Baseline Models (YOLOv8s, YOLOv5nu, YOLOv8n-scratch)')
    
    # Phase 9: Ablation Study
    run_step(['run_ablation.py'], 'Phase 9: 2x2 Grid Ablation Study')
    
    # Phase 11: Paper Numbers & Evidence Compilation
    run_step(['generate_paper_numbers.py'], 'Phase 11: Generate Paper Numbers Document')
    
    print("\n" + "=" * 60)
    print("ALL PIPELINE PHASES (4 - 11) COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    main()
