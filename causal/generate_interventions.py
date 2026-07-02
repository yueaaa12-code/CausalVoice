"""
Generate real causal intervention data for CausalVoice training.

Creates intervention groups from real audio using torchaudio transforms:
- Speaker change (HIGH causal effect): same text, different speaker
- Noise addition (LOW causal effect): same audio + noise/reverb
- Speed change (MEDIUM causal effect): time-stretch

Each group: {anchor, interventions with causal_effect labels}
Output: data/synthetic_causal/{train,test}/group_XXXX.pt

Usage:
    python causal/generate_interventions.py \
        --source_dir data/vctk/wav48_silence_trimmed \
        --output_dir data/synthetic_causal/train \
        --num_groups 5000
"""

import os
import argparse
import random
import glob
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm


def add_noise(wav, snr_db=20):
    """Add white noise at given SNR."""
    noise = torch.randn_like(wav)
    signal_power = wav.pow(2).mean()
    noise_power = noise.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-8))
    return wav + noise * scale


def add_reverb(wav, sr=16000):
    """Add simple synthetic reverb."""
    delay_samples = int(sr * 0.03)
    decay = 0.3
    reverbed = wav.clone()
    if wav.size(-1) > delay_samples:
        reverbed[..., delay_samples:] += decay * wav[..., :-delay_samples]
    return reverbed


def change_speed(wav, sr=16000, factor=1.0):
    """Change speed via resampling."""
    if abs(factor - 1.0) < 0.01:
        return wav
    new_sr = int(sr * factor)
    resampled = torchaudio.functional.resample(wav, sr, new_sr)
    result = torchaudio.functional.resample(resampled, new_sr, sr)
    return result


def build_speaker_groups(source_dir):
    """Scan VCTK-style directory and group files by sentence."""
    speakers = sorted([d for d in os.listdir(source_dir)
                       if os.path.isdir(os.path.join(source_dir, d))])

    sentence_map = {}
    for spk in speakers:
        spk_dir = os.path.join(source_dir, spk)
        for f in glob.glob(os.path.join(spk_dir, "*.flac")) + \
                 glob.glob(os.path.join(spk_dir, "*.wav")):
            basename = os.path.basename(f)
            parts = basename.replace('.flac', '').replace('.wav', '').split('_')
            if len(parts) >= 2:
                sent_id = parts[1]
            else:
                sent_id = parts[0]
            if sent_id not in sentence_map:
                sentence_map[sent_id] = {}
            sentence_map[sent_id][spk] = f

    multi_speaker = {k: v for k, v in sentence_map.items() if len(v) >= 3}
    return multi_speaker, speakers


def generate_groups(args):
    """Main generation pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load SSL model
    print("Loading wav2vec2-base for feature extraction...")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    ssl_model = bundle.get_model().to(device).eval()
    ssl_dim = 768
    target_sr = bundle.sample_rate

    # Scan source
    has_real_audio = args.source_dir and os.path.exists(args.source_dir)
    if has_real_audio:
        print(f"Scanning: {args.source_dir}")
        sentence_map, speakers = build_speaker_groups(args.source_dir)
        sentence_ids = list(sentence_map.keys())
        print(f"  {len(sentence_map)} sentences, {len(speakers)} speakers")
    else:
        sentence_ids = []
        print("No source_dir — using synthetic fallback.")

    print(f"Generating {args.num_groups} groups...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    generated = 0

    for group_idx in tqdm(range(args.num_groups)):
        group_path = output_dir / f"group_{group_idx:05d}.pt"
        if group_path.exists():
            generated += 1
            continue

        items = []       # list of (T, dim) tensors
        causal_effects = []

        if has_real_audio and sentence_ids:
            sent_id = random.choice(sentence_ids)
            spk_files = sentence_map[sent_id]
            available = list(spk_files.keys())
            if len(available) < 2:
                continue

            anchor_spk = random.choice(available)
            anchor_wav, sr = torchaudio.load(spk_files[anchor_spk])
            if sr != target_sr:
                anchor_wav = torchaudio.functional.resample(anchor_wav, sr, target_sr)
            if anchor_wav.size(0) > 1:
                anchor_wav = anchor_wav.mean(0, keepdim=True)

            with torch.no_grad():
                feats, _ = ssl_model.extract_features(anchor_wav.to(device))
                anchor_feat = feats[-1].squeeze(0).cpu()
            items.append(anchor_feat)

            # Speaker changes (HIGH CE = 0.6-0.8)
            others = [s for s in available if s != anchor_spk]
            for spk in random.sample(others, min(args.n_speaker, len(others))):
                w, s = torchaudio.load(spk_files[spk])
                if s != target_sr:
                    w = torchaudio.functional.resample(w, s, target_sr)
                if w.size(0) > 1:
                    w = w.mean(0, keepdim=True)
                with torch.no_grad():
                    f, _ = ssl_model.extract_features(w.to(device))
                    items.append(f[-1].squeeze(0).cpu())
                causal_effects.append(0.7 + random.uniform(-0.1, 0.1))

            # Noise (LOW CE = 0.01-0.04)
            for snr in random.sample([10, 15, 20, 25, 30], min(args.n_noise, 5)):
                noisy = add_noise(anchor_wav, snr)
                with torch.no_grad():
                    f, _ = ssl_model.extract_features(noisy.to(device))
                    items.append(f[-1].squeeze(0).cpu())
                causal_effects.append(0.02 + random.uniform(0, 0.02))

            # Reverb (LOW CE)
            rev = add_reverb(anchor_wav, target_sr)
            with torch.no_grad():
                f, _ = ssl_model.extract_features(rev.to(device))
                items.append(f[-1].squeeze(0).cpu())
            causal_effects.append(0.03)

            # Speed (MEDIUM CE = 0.2-0.4)
            for factor in random.sample([0.85, 0.9, 1.1, 1.15], min(args.n_speed, 4)):
                sp = change_speed(anchor_wav, target_sr, factor)
                with torch.no_grad():
                    f, _ = ssl_model.extract_features(sp.to(device))
                    items.append(f[-1].squeeze(0).cpu())
                causal_effects.append(0.3 + random.uniform(-0.1, 0.1))

        else:
            # Synthetic fallback
            T = random.randint(30, 100)
            anchor_feat = torch.randn(T, ssl_dim) * 0.5
            items.append(anchor_feat)
            for _ in range(args.n_speaker):
                d = torch.randn(1, ssl_dim) * 0.3
                items.append(anchor_feat + d.expand(T, -1) + torch.randn(T, ssl_dim) * 0.05)
                causal_effects.append(0.7 + random.uniform(-0.1, 0.1))
            for _ in range(args.n_noise):
                items.append(anchor_feat + torch.randn(T, ssl_dim) * 1.2)
                causal_effects.append(0.02 + random.uniform(0, 0.03))
            for _ in range(args.n_speed):
                items.append(anchor_feat + torch.randn(T, ssl_dim) * 0.5)
                causal_effects.append(0.3 + random.uniform(-0.1, 0.1))

        torch.save({
            'items': items,
            'causal_effects': torch.tensor(causal_effects, dtype=torch.float32),
        }, group_path)
        generated += 1

    print(f"Done: {generated} groups in {output_dir}")
    torch.save({'num_groups': generated, 'ssl_dim': ssl_dim, 'seed': args.seed},
               output_dir / 'metadata.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_groups', type=int, default=5000)
    parser.add_argument('--n_speaker', type=int, default=3)
    parser.add_argument('--n_noise', type=int, default=2)
    parser.add_argument('--n_speed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate_groups(args)
