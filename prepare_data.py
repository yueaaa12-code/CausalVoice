"""
Prepare data: create filelists and extract F0 from VCTK dataset.

Usage:
    python prepare_data.py [--vctk_dir data/vctk/wav48_silence_trimmed] [--n_speakers 20]
"""

import os
import glob
import random
import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm


def prepare_filelists(vctk_dir, n_speakers=20):
    """Create train/val filelists from VCTK."""
    speakers = sorted([d for d in os.listdir(vctk_dir)
                       if os.path.isdir(os.path.join(vctk_dir, d))])[:n_speakers]
    
    train_lines, val_lines = [], []
    for spk in speakers:
        spk_dir = os.path.join(vctk_dir, spk)
        wavs = sorted(glob.glob(f"{spk_dir}/*.flac") + glob.glob(f"{spk_dir}/*.wav"))
        random.shuffle(wavs)
        split = max(1, int(len(wavs) * 0.9))
        for w in wavs[:split]:
            train_lines.append(f"{w}|{spk}\n")
        for w in wavs[split:]:
            val_lines.append(f"{w}|{spk}\n")
    
    os.makedirs("filelists", exist_ok=True)
    with open("filelists/train_real.txt", "w") as f:
        f.writelines(train_lines)
    with open("filelists/val_real.txt", "w") as f:
        f.writelines(val_lines)
    
    print(f"Created filelists: train={len(train_lines)}, val={len(val_lines)}, speakers={len(speakers)}")
    return train_lines + val_lines


def extract_f0(filelist_lines, f0_dir="data/f0"):
    """Extract F0 for all audio files."""
    os.makedirs(f0_dir, exist_ok=True)
    
    for line in tqdm(filelist_lines, desc="Extracting F0"):
        wav_path = line.strip().split("|")[0]
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        f0_path = os.path.join(f0_dir, f"{basename}.pt")
        
        if os.path.exists(f0_path):
            continue
        
        try:
            wav, sr = torchaudio.load(wav_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)
            
            # Simple F0 extraction
            f0 = torchaudio.functional.detect_pitch_frequency(wav, 16000, freq_low=50, freq_high=600)
            # Downsample to match spectrogram frames (hop=320 → 50fps)
            n_frames = wav.size(-1) // 320
            if f0.size(-1) > n_frames:
                f0 = f0[:, :n_frames]
            
            torch.save(f0, f0_path)
        except Exception as e:
            # Save zeros as fallback
            n_frames = 100
            torch.save(torch.zeros(1, n_frames), f0_path)


def extract_ssl_features(filelist_lines, feature_dir="data/wavlm_features"):
    """Extract wav2vec2/WavLM features."""
    os.makedirs(feature_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(device).eval()
    
    for line in tqdm(filelist_lines, desc="Extracting SSL features"):
        wav_path = line.strip().split("|")[0]
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        feat_path = os.path.join(feature_dir, f"{basename}.pt")
        
        if os.path.exists(feat_path):
            continue
        
        try:
            wav, sr = torchaudio.load(wav_path)
            if sr != bundle.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)
            
            with torch.no_grad():
                features, _ = model.extract_features(wav.to(device))
                feat = features[-1].squeeze(0).cpu()  # (T, 768)
            
            torch.save(feat, feat_path)
        except Exception as e:
            print(f"  Error: {wav_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vctk_dir', type=str, default='data/vctk/wav48_silence_trimmed')
    parser.add_argument('--n_speakers', type=int, default=20)
    parser.add_argument('--feature_dir', type=str, default='data/wavlm_features')
    parser.add_argument('--f0_dir', type=str, default='data/f0')
    args = parser.parse_args()
    
    lines = prepare_filelists(args.vctk_dir, args.n_speakers)
    extract_f0(lines, args.f0_dir)
    extract_ssl_features(lines, args.feature_dir)
    print("Data preparation complete!")
