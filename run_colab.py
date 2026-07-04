#!/usr/bin/env python3
"""
CausalVoice: One-click Colab Pipeline
Run this script on Colab (A100 or T4 GPU) to get complete comparison results.

Usage:
    !python run_colab.py --stage all
    !python run_colab.py --stage all --n_speakers 10 --quick  # fast test run
"""

import os
import sys
import subprocess
import argparse
import time
import json


def run(cmd, check=True):
    """Run shell command with output."""
    print(f"\n{'='*60}")
    print(f"$ {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0


def stage_setup():
    """Install dependencies and verify GPU."""
    print("\n" + "★" * 60)
    print("STAGE 1: SETUP")
    print("★" * 60)
    
    # Colab already has torch; install additional deps
    run("pip install -q librosa soundfile tensorboard tqdm matplotlib pesq pystoi")
    
    import torch
    assert torch.cuda.is_available(), "ERROR: No GPU detected! Select GPU runtime in Colab."
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA: {torch.version.cuda}")
    
    # Recommend batch size based on GPU
    if 'A100' in gpu_name:
        print("✓ A100 detected → batch_size=16, fp16=true")
        return {'batch_size': 16, 'fp16': True}
    else:
        print("✓ T4 detected → batch_size=8, fp16=true")
        return {'batch_size': 8, 'fp16': True}


def stage_data(n_speakers=20):
    """Download VCTK and prepare all data."""
    print("\n" + "★" * 60)
    print("STAGE 2: DATA PREPARATION")
    print("★" * 60)
    
    vctk_wav_dir = "data/vctk/wav48_silence_trimmed"
    
    # Step 1: Download VCTK
    if not os.path.exists(vctk_wav_dir):
        print("Downloading VCTK corpus (~10GB, takes 5-10 min)...")
        run("mkdir -p data")
        run("wget -q --show-progress "
            "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip "
            "-O data/vctk.zip")
        run("unzip -q data/vctk.zip -d data/vctk")
        run("rm data/vctk.zip")
    else:
        print(f"✓ VCTK already exists at {vctk_wav_dir}")
    
    # Step 2: Create filelists + extract F0 + SSL features
    print("\nPreparing filelists + F0 + SSL features...")
    run(f"python prepare_data.py "
        f"--vctk_dir {vctk_wav_dir} "
        f"--n_speakers {n_speakers} "
        f"--feature_dir data/wavlm_features "
        f"--f0_dir data/f0")
    
    # Step 3: Update n_speakers in config to match actual data
    config = _load_config()
    config['model']['n_speakers'] = n_speakers
    _save_config(config, 'configs/causal_vc.json')
    
    # Step 4: Generate causal intervention data
    causal_train_dir = "data/causal_interventions"
    causal_test_dir = "data/causal_interventions_test"
    
    if not os.path.exists(os.path.join(causal_train_dir, "metadata.pt")):
        print("\nGenerating causal intervention data (train)...")
        run(f"python causal/generate_interventions.py "
            f"--source_dir {vctk_wav_dir} "
            f"--output_dir {causal_train_dir} "
            f"--num_groups 3000 --seed 42")
    else:
        print(f"✓ Causal train data exists at {causal_train_dir}")
    
    if not os.path.exists(os.path.join(causal_test_dir, "metadata.pt")):
        print("\nGenerating causal intervention data (test)...")
        run(f"python causal/generate_interventions.py "
            f"--source_dir {vctk_wav_dir} "
            f"--output_dir {causal_test_dir} "
            f"--num_groups 300 --seed 123")
    else:
        print(f"✓ Causal test data exists at {causal_test_dir}")
    
    # Verify
    import glob as g
    n_train = len(g.glob(f"{causal_train_dir}/group_*.pt"))
    n_test = len(g.glob(f"{causal_test_dir}/group_*.pt"))
    print(f"\n✓ Data ready: {n_train} train groups, {n_test} test groups")
    print(f"✓ Filelists: filelists/train_real.txt, filelists/val_real.txt")
    print(f"✓ n_speakers={n_speakers} (updated in config)")


def stage_train_baseline(epochs=200, batch_size=16, fp16=True):
    """Train baseline O²-VC (no causal regularization)."""
    print("\n" + "★" * 60)
    print("STAGE 3: TRAIN BASELINE (O²-VC, no causal)")
    print("★" * 60)
    
    # Update config for baseline: same architecture, no causal loss
    config = _load_config()
    config['train']['batch_size'] = batch_size
    config['train']['epochs'] = epochs
    config['train']['fp16_run'] = fp16
    config['causal']['reg_type'] = 'none'  # KEY: no causal regularization
    config['causal']['data_path'] = 'data/causal_interventions'
    config['causal']['weight_path'] = ''
    _save_config(config, 'configs/baseline.json')
    
    run("python train_causal.py -c configs/baseline.json -m logs/baseline")


def stage_train_causal(epochs=200, batch_size=16, fp16=True):
    """Train CausalVoice (with causal regularization). Same epochs as baseline for fair comparison."""
    print("\n" + "★" * 60)
    print("STAGE 4: TRAIN CAUSALVOICE (with causal)")
    print("★" * 60)
    
    # FAIR COMPARISON: same total epochs, same init (random), only difference is causal loss
    config = _load_config()
    config['train']['batch_size'] = batch_size
    config['train']['epochs'] = epochs
    config['train']['fp16_run'] = fp16
    config['causal']['reg_type'] = 'both'  # KEY: causal regularization ON
    config['causal']['data_path'] = 'data/causal_interventions'
    config['causal']['weight_path'] = ''   # NO pretrained init — train from scratch
    config['causal']['start_epoch'] = epochs // 4  # Warm up 25% before adding causal loss
    _save_config(config, 'configs/causal.json')
    
    print(f"  Causal loss activates at epoch {config['causal']['start_epoch']}/{epochs}")
    print(f"  Same random init as baseline (fair comparison)")
    
    run("python train_causal.py -c configs/causal.json -m logs/causal_voice")


def stage_evaluate():
    """Run full evaluation pipeline comparing baseline vs CausalVoice."""
    print("\n" + "★" * 60)
    print("STAGE 5: EVALUATE & COMPARE")
    print("★" * 60)
    
    import glob
    baseline_ckpts = sorted(glob.glob("logs/baseline/G_*.pth"))
    causal_ckpts = sorted(glob.glob("logs/causal_voice/G_*.pth"))
    
    if not baseline_ckpts:
        print("ERROR: No baseline checkpoints found. Run train_baseline first!")
        return
    if not causal_ckpts:
        print("ERROR: No causal checkpoints found. Run train_causal first!")
        return
    
    baseline_best = baseline_ckpts[-1]
    causal_best = causal_ckpts[-1]
    
    print(f"Baseline checkpoint: {baseline_best}")
    print(f"CausalVoice checkpoint: {causal_best}")
    
    run(f"python evaluate_pipeline.py "
        f"--baseline_ckpt {baseline_best} "
        f"--causal_ckpt {causal_best} "
        f"--test_filelist filelists/val_real.txt "
        f"--causal_test_path data/causal_interventions_test "
        f"--config configs/causal_vc.json "
        f"--output_dir results/ "
        f"--max_samples 50")
    
    # Print results
    results_file = "results/comparison.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print("\n" + "=" * 60)
        print("FINAL RESULTS: Baseline (O²-VC) vs CausalVoice")
        print("=" * 60)
        for metric, values in results.items():
            if isinstance(values, dict):
                baseline_val = values.get('baseline', 'N/A')
                causal_val = values.get('causal', 'N/A')
                print(f"  {metric:30s}: {baseline_val:>10} → {causal_val:>10}")


def stage_quick_verify():
    """Quick verification with synthetic data (no real data needed)."""
    print("\n" + "★" * 60)
    print("QUICK VERIFY: Testing causal mechanism")
    print("★" * 60)
    
    run("python verify_feasibility.py")


# ======================== HELPERS ========================

def _load_config(path='configs/causal_vc.json'):
    with open(path) as f:
        return json.load(f)

def _save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="CausalVoice Colab Pipeline")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'setup', 'verify', 'data',
                                 'train_baseline', 'train_causal', 'evaluate'])
    parser.add_argument('--n_speakers', type=int, default=20,
                        help='Number of VCTK speakers to use (default: 20)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs (default: 200)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 10 speakers, 50 epochs')
    args = parser.parse_args()
    
    if args.quick:
        args.n_speakers = 10
        args.epochs = 50
    
    gpu_config = {'batch_size': 16, 'fp16': True}
    
    if args.stage == 'all':
        gpu_config = stage_setup() or gpu_config
        stage_data(args.n_speakers)
        stage_train_baseline(args.epochs, **gpu_config)
        stage_train_causal(args.epochs, **gpu_config)
        stage_evaluate()
    elif args.stage == 'setup':
        stage_setup()
    elif args.stage == 'verify':
        stage_quick_verify()
    elif args.stage == 'data':
        stage_data(args.n_speakers)
    elif args.stage == 'train_baseline':
        stage_train_baseline(args.epochs)
    elif args.stage == 'train_causal':
        stage_train_causal(args.epochs)
    elif args.stage == 'evaluate':
        stage_evaluate()
    
    print("\n" + "★" * 60)
    print("PIPELINE COMPLETE!")
    print("★" * 60)


if __name__ == '__main__':
    main()
