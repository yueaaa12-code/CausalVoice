"""
CausalVoice Full Pipeline - Breakpoint Resumable
=================================================
Designed for Google Colab. Saves progress to Google Drive after each stage.
If runtime disconnects, re-run this script and it resumes from last completed stage.

Usage in Colab:
    Cell 1: from google.colab import drive; drive.mount('/content/drive')
    Cell 2: !git clone https://github.com/yueaaa12-code/CausalVoice.git /content/CausalVoice
    Cell 3: %cd /content/CausalVoice && python run_full_pipeline.py
"""

import os
import sys
import json
import shutil
import subprocess

# ============================================================
# CONFIG
# ============================================================
DRIVE_DIR = "/content/drive/MyDrive/CausalVoice_results"
STATE_FILE = os.path.join(DRIVE_DIR, "pipeline_state.json")
N_SPEAKERS = 20
EPOCHS = 100
CAUSAL_START_EPOCH = 25

# ============================================================
# UTILS
# ============================================================

def run(cmd, check=True):
    print(f"\n{'='*60}\n$ {cmd}\n{'='*60}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed_stages": []}

def save_state(state):
    os.makedirs(DRIVE_DIR, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def stage_done(state, stage_name):
    return stage_name in state["completed_stages"]

def mark_done(state, stage_name):
    state["completed_stages"].append(stage_name)
    save_state(state)
    print(f"\n✓ Stage '{stage_name}' completed and saved to Drive.\n")

# ============================================================
# STAGE 1: SETUP
# ============================================================
def stage_setup(state):
    if stage_done(state, "setup"):
        print("✓ Setup already done, skipping.")
        return
    
    run("pip install -q librosa soundfile tensorboard tqdm matplotlib pesq pystoi")
    
    # Fix known issues
    run("sed -i 's/total_mem/total_memory/' run_colab.py", check=False)
    run("sed -i 's/spec = torch.stft(y, n_fft/spec = torch.stft(y.float(), n_fft/' mel_processing.py", check=False)
    
    # Detect GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠ No GPU detected, training will be slow!")
    
    mark_done(state, "setup")

# ============================================================
# STAGE 2: DATA DOWNLOAD
# ============================================================
def stage_data_download(state):
    if stage_done(state, "data_download"):
        print("✓ Data download already done, skipping.")
        return
    
    # Check if data exists on Drive (from previous run)
    drive_data_marker = os.path.join(DRIVE_DIR, "data_ready.flag")
    local_data_dir = "data/vctk/wav48_silence_trimmed"
    
    if os.path.exists(drive_data_marker) and os.path.exists(os.path.join(DRIVE_DIR, "wav48_silence_trimmed")):
        print("Restoring data from Drive...")
        os.makedirs("data/vctk", exist_ok=True)
        if not os.path.exists(local_data_dir):
            shutil.copytree(os.path.join(DRIVE_DIR, "wav48_silence_trimmed"), local_data_dir)
        mark_done(state, "data_download")
        return
    
    # Download LibriSpeech
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/LibriSpeech/train-clean-100"):
        run("wget -q --show-progress https://www.openslr.org/resources/12/train-clean-100.tar.gz -O data/libri.tar.gz")
        run("tar -xzf data/libri.tar.gz -C data/")
        run("rm -f data/libri.tar.gz")
    
    # Reorganize to VCTK structure
    src = "data/LibriSpeech/train-clean-100"
    os.makedirs(local_data_dir, exist_ok=True)
    speakers = sorted(os.listdir(src))[:N_SPEAKERS]
    for spk in speakers:
        spk_dst = os.path.join(local_data_dir, spk)
        os.makedirs(spk_dst, exist_ok=True)
        for chapter in os.listdir(os.path.join(src, spk)):
            chapter_dir = os.path.join(src, spk, chapter)
            if not os.path.isdir(chapter_dir):
                continue
            for f in os.listdir(chapter_dir):
                if f.endswith('.flac'):
                    dst_path = os.path.join(spk_dst, f)
                    if not os.path.exists(dst_path):
                        shutil.copy2(os.path.join(chapter_dir, f), dst_path)
        print(f"  {spk}: {len(os.listdir(spk_dst))} utterances")
    
    # Save to Drive for future recovery
    print("Saving data to Drive (for recovery)...")
    drive_data_dst = os.path.join(DRIVE_DIR, "wav48_silence_trimmed")
    if not os.path.exists(drive_data_dst):
        shutil.copytree(local_data_dir, drive_data_dst)
    with open(drive_data_marker, 'w') as f:
        f.write("done")
    
    mark_done(state, "data_download")

# ============================================================
# STAGE 3: FEATURE EXTRACTION
# ============================================================
def stage_features(state):
    if stage_done(state, "features"):
        print("✓ Feature extraction already done, skipping.")
        return
    
    # Check Drive for cached features
    drive_features = os.path.join(DRIVE_DIR, "features_cache")
    if os.path.exists(os.path.join(drive_features, "done.flag")):
        print("Restoring features from Drive...")
        if os.path.exists(os.path.join(drive_features, "wavlm_features")):
            shutil.copytree(os.path.join(drive_features, "wavlm_features"), "data/wavlm_features", dirs_exist_ok=True)
        if os.path.exists(os.path.join(drive_features, "f0")):
            shutil.copytree(os.path.join(drive_features, "f0"), "data/f0", dirs_exist_ok=True)
        if os.path.exists(os.path.join(drive_features, "filelists")):
            shutil.copytree(os.path.join(drive_features, "filelists"), "filelists", dirs_exist_ok=True)
        if os.path.exists(os.path.join(drive_features, "causal_interventions")):
            shutil.copytree(os.path.join(drive_features, "causal_interventions"), "data/causal_interventions", dirs_exist_ok=True)
        mark_done(state, "features")
        return
    
    run(f"python prepare_data.py --vctk_dir data/vctk/wav48_silence_trimmed --n_speakers {N_SPEAKERS}")
    run("python causal/generate_interventions.py --source_dir data/vctk/wav48_silence_trimmed --output_dir data/causal_interventions")
    
    # Cache to Drive
    print("Caching features to Drive...")
    os.makedirs(drive_features, exist_ok=True)
    for src_dir, name in [("data/wavlm_features", "wavlm_features"), 
                           ("data/f0", "f0"),
                           ("filelists", "filelists"),
                           ("data/causal_interventions", "causal_interventions")]:
        if os.path.exists(src_dir):
            dst = os.path.join(drive_features, name)
            if not os.path.exists(dst):
                shutil.copytree(src_dir, dst)
    with open(os.path.join(drive_features, "done.flag"), 'w') as f:
        f.write("done")
    
    mark_done(state, "features")

# ============================================================
# STAGE 4: TRAIN BASELINE
# ============================================================
def stage_train_baseline(state):
    if stage_done(state, "train_baseline"):
        print("✓ Baseline training already done, skipping.")
        # Restore checkpoint if needed
        drive_ckpt = os.path.join(DRIVE_DIR, "baseline")
        if os.path.exists(drive_ckpt) and not os.path.exists("logs/baseline"):
            shutil.copytree(drive_ckpt, "logs/baseline")
        return
    
    # Create config
    cfg = json.load(open('configs/causal_vc.json'))
    cfg['train']['epochs'] = EPOCHS
    cfg['causal']['reg_type'] = 'none'
    cfg['model_dir'] = 'logs/baseline'
    with open('configs/baseline.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    
    print(f"Training baseline: {EPOCHS} epochs, no causal regularization")
    run("python train_causal.py -c configs/baseline.json -m logs/baseline")
    
    # Save to Drive
    if os.path.exists("logs/baseline"):
        drive_dst = os.path.join(DRIVE_DIR, "baseline")
        if os.path.exists(drive_dst):
            shutil.rmtree(drive_dst)
        shutil.copytree("logs/baseline", drive_dst)
    
    mark_done(state, "train_baseline")

# ============================================================
# STAGE 5: TRAIN CAUSALVOICE
# ============================================================
def stage_train_causal(state):
    if stage_done(state, "train_causal"):
        print("✓ CausalVoice training already done, skipping.")
        drive_ckpt = os.path.join(DRIVE_DIR, "causal")
        if os.path.exists(drive_ckpt) and not os.path.exists("logs/causal"):
            shutil.copytree(drive_ckpt, "logs/causal")
        return
    
    # Create config
    cfg = json.load(open('configs/causal_vc.json'))
    cfg['train']['epochs'] = EPOCHS
    cfg['causal']['reg_type'] = 'both'
    cfg['causal']['start_epoch'] = CAUSAL_START_EPOCH
    cfg['model_dir'] = 'logs/causal'
    with open('configs/causal.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    
    print(f"Training CausalVoice: {EPOCHS} epochs, causal starts at epoch {CAUSAL_START_EPOCH}")
    run("python train_causal.py -c configs/causal.json -m logs/causal")
    
    # Save to Drive
    if os.path.exists("logs/causal"):
        drive_dst = os.path.join(DRIVE_DIR, "causal")
        if os.path.exists(drive_dst):
            shutil.rmtree(drive_dst)
        shutil.copytree("logs/causal", drive_dst)
    
    mark_done(state, "train_causal")

# ============================================================
# STAGE 5b: ABLATION STUDY (contrastive-only, ranking-only)
# ============================================================
def stage_ablation(state):
    if stage_done(state, "ablation"):
        print("✓ Ablation training already done, skipping.")
        for name in ["ablation_contrastive", "ablation_ranking"]:
            drive_ckpt = os.path.join(DRIVE_DIR, name)
            if os.path.exists(drive_ckpt) and not os.path.exists(f"logs/{name}"):
                shutil.copytree(drive_ckpt, f"logs/{name}")
        return
    
    ablation_configs = [
        ("ablation_contrastive", "contrastive"),
        ("ablation_ranking", "ranking"),
    ]
    
    for name, reg_type in ablation_configs:
        cfg = json.load(open('configs/causal_vc.json'))
        cfg['train']['epochs'] = EPOCHS
        cfg['causal']['reg_type'] = reg_type
        cfg['causal']['start_epoch'] = CAUSAL_START_EPOCH
        cfg['model_dir'] = f'logs/{name}'
        config_path = f'configs/{name}.json'
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        
        print(f"Ablation: {name} (reg_type={reg_type}), {EPOCHS} epochs")
        run(f"python train_causal.py -c {config_path} -m logs/{name}")
        
        if os.path.exists(f"logs/{name}"):
            drive_dst = os.path.join(DRIVE_DIR, name)
            if os.path.exists(drive_dst):
                shutil.rmtree(drive_dst)
            shutil.copytree(f"logs/{name}", drive_dst)
    
    mark_done(state, "ablation")

# ============================================================
# STAGE 6: EVALUATE
# ============================================================
def stage_evaluate(state):
    if stage_done(state, "evaluate"):
        print("✓ Evaluation already done.")
        results_path = os.path.join(DRIVE_DIR, "comparison_results.json")
        if os.path.exists(results_path):
            print("\n" + "="*60)
            print("RESULTS:")
            print("="*60)
            print(open(results_path).read())
        return
    
    # Ensure checkpoints exist locally
    for name in ["baseline", "causal", "ablation_contrastive", "ablation_ranking"]:
        local_dir = f"logs/{name}"
        if not os.path.exists(local_dir):
            drive_src = os.path.join(DRIVE_DIR, name)
            if os.path.exists(drive_src):
                shutil.copytree(drive_src, local_dir)
    
    # Find best checkpoint (highest step number)
    def find_best_ckpt(model_dir):
        import glob as g
        ckpts = g.glob(os.path.join(model_dir, "G_*.pth"))
        if not ckpts:
            return None
        return max(ckpts, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    baseline_ckpt = find_best_ckpt("logs/baseline")
    causal_ckpt = find_best_ckpt("logs/causal")
    
    if not baseline_ckpt or not causal_ckpt:
        print("ERROR: Missing checkpoints!")
        return
    
    # Build eval command
    cmd = (f"python evaluate_pipeline.py "
           f"--baseline_ckpt {baseline_ckpt} "
           f"--causal_ckpt {causal_ckpt} "
           f"--config configs/causal_vc.json "
           f"--causal_test_path data/causal_interventions "
           f"--test_filelist filelists/val_real.txt "
           f"--train_filelist filelists/train_real.txt")
    
    # Add ablation models if available
    ablation_ckpts = []
    ablation_names = []
    for name in ["ablation_contrastive", "ablation_ranking"]:
        ckpt = find_best_ckpt(f"logs/{name}")
        if ckpt:
            ablation_ckpts.append(ckpt)
            ablation_names.append(name)
    
    if ablation_ckpts:
        cmd += f" --ablation_ckpts {','.join(ablation_ckpts)}"
        cmd += f" --ablation_names {','.join(ablation_names)}"
    
    run(cmd)
    
    # Save results to Drive
    if os.path.exists("results"):
        for f in os.listdir("results"):
            shutil.copy2(os.path.join("results", f), DRIVE_DIR)
    
    mark_done(state, "evaluate")
    
    # Print results
    results_file = "results/comparison_results.json"
    if os.path.exists(results_file):
        print("\n" + "★"*60)
        print("FINAL RESULTS:")
        print("★"*60)
        print(open(results_file).read())

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("CausalVoice Pipeline - Breakpoint Resumable")
    print("="*60)
    
    # Verify Drive is mounted
    if not os.path.exists("/content/drive/MyDrive"):
        print("ERROR: Google Drive not mounted!")
        print("Run this first: from google.colab import drive; drive.mount('/content/drive')")
        sys.exit(1)
    
    os.makedirs(DRIVE_DIR, exist_ok=True)
    state = load_state()
    
    print(f"Completed stages: {state['completed_stages']}")
    print(f"Config: {N_SPEAKERS} speakers, {EPOCHS} epochs")
    print()
    
    stages = [
        ("setup", stage_setup),
        ("data_download", stage_data_download),
        ("features", stage_features),
        ("train_baseline", stage_train_baseline),
        ("train_causal", stage_train_causal),
        ("ablation", stage_ablation),
        ("evaluate", stage_evaluate),
    ]
    
    for name, func in stages:
        print(f"\n{'★'*60}")
        print(f"STAGE: {name.upper()}")
        print(f"{'★'*60}")
        func(state)
    
    print("\n" + "★"*60)
    print("ALL STAGES COMPLETE!")
    print(f"Results saved to: {DRIVE_DIR}")
    print("★"*60)

if __name__ == "__main__":
    main()
