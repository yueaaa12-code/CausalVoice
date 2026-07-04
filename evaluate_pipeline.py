"""
Full evaluation pipeline for CausalVoice vs Baseline comparison.

Produces all metrics needed for paper:
1. Causal awareness: HNC, ARS, DR
2. VC quality: Speaker Similarity (wav2vec cosine), MCD
3. Noise robustness: quality degradation under noise
4. Cross-domain generalization: unseen speaker performance
5. Ablation study support: contrastive-only, ranking-only, both

Usage:
    python evaluate_pipeline.py \
        --baseline_ckpt logs/baseline/G_12000.pth \
        --causal_ckpt logs/causal/G_12000.pth \
        --test_filelist filelists/val_real.txt \
        --causal_test_path data/causal_interventions \
        --config configs/causal_vc.json \
        --output_dir results/

    # With ablation:
    python evaluate_pipeline.py \
        --baseline_ckpt logs/baseline/G_12000.pth \
        --causal_ckpt logs/causal/G_12000.pth \
        --ablation_ckpts logs/ablation_contrastive/G_12000.pth,logs/ablation_ranking/G_12000.pth \
        --ablation_names contrastive_only,ranking_only \
        --config configs/causal_vc.json
"""

import os
import json
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models_causal import SynthesizerTrnCausal
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from causal.losses import calc_contrastive_loss, calc_ranking_loss
from causal.dataset import CausalInterventionDataset, causal_collate_fn
from convert import load_model, extract_ssl_features, extract_f0
import utils


# ============================================================
# Metric 1: Causal Awareness (HNC / ARS / DR)
# ============================================================

def evaluate_causal_awareness(model, causal_test_path, config, device='cuda'):
    """Evaluate causal awareness metrics on held-out intervention data."""
    
    dataset = CausalInterventionDataset(causal_test_path, config)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, 
        collate_fn=causal_collate_fn, drop_last=False
    )
    
    all_hnc, all_ars = 0, 0
    all_high_dists, all_low_dists = [], []
    total_groups = 0
    rank_correct, rank_total = 0, 0
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features, causal_effects, data_splits = batch
            features = features.to(device)
            
            embeds = model.get_causal_embeddings(features, c_lengths=None)
            
            for g_idx in range(len(causal_effects)):
                anchor = embeds[data_splits[g_idx]]
                keys = embeds[data_splits[g_idx]+1:data_splits[g_idx+1]]
                ce = causal_effects[g_idx].numpy()
                
                # Cosine distances
                dists = (1.0 - torch.matmul(keys, anchor)).cpu().numpy()
                
                # HNC: non-causal items should have low distance
                low_ce_mask = ce < 0.05
                high_ce_mask = ce > 0.3
                
                if low_ce_mask.any():
                    all_low_dists.extend(dists[low_ce_mask].tolist())
                    all_hnc += (dists[low_ce_mask] > 0.1).sum()
                
                if high_ce_mask.any():
                    all_high_dists.extend(dists[high_ce_mask].tolist())
                
                # Pairwise ranking
                for i in range(len(ce)):
                    for j in range(i+1, len(ce)):
                        if abs(ce[i] - ce[j]) > 0.1:
                            rank_total += 1
                            if (ce[i] > ce[j]) == (dists[i] > dists[j]):
                                rank_correct += 1
                
                total_groups += 1
    
    hnc = 1.0 - (all_hnc / max(len(all_low_dists), 1))
    ars = rank_correct / max(rank_total, 1)
    dr = np.mean(all_high_dists) / max(np.mean(all_low_dists), 1e-8) if all_low_dists else 0
    
    return {
        'HNC': hnc,
        'ARS': ars,
        'DR': dr,
        'mean_high_ce_dist': np.mean(all_high_dists) if all_high_dists else 0,
        'mean_low_ce_dist': np.mean(all_low_dists) if all_low_dists else 0,
        'n_groups': total_groups,
    }


# ============================================================
# Metric 2: VC Quality (Speaker Similarity, MCD)
# ============================================================

def mel_cepstral_distortion(mel_ref, mel_gen):
    """Compute MCD between two mel spectrograms."""
    # Align lengths
    min_len = min(mel_ref.size(-1), mel_gen.size(-1))
    mel_ref = mel_ref[..., :min_len]
    mel_gen = mel_gen[..., :min_len]
    
    diff = mel_ref - mel_gen
    mcd = torch.sqrt(2 * torch.sum(diff ** 2, dim=-2)).mean()
    return mcd.item()


def speaker_similarity_cosine(emb1, emb2):
    """Cosine similarity between speaker embeddings."""
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).mean().item()


def get_speaker_embedding_from_wav(wav, sr, ssl_model, device):
    """Extract speaker embedding from audio using wav2vec2 mean pooling."""
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    with torch.no_grad():
        feats, _ = ssl_model.extract_features(wav.to(device))
        # Use last layer, mean pool over time → speaker embedding
        emb = feats[-1].squeeze(0).mean(dim=0)  # (768,)
    return emb


def evaluate_vc_quality(model, test_filelist, hps, device='cuda', max_samples=50):
    """
    Evaluate VC quality: convert source → target speaker, measure quality.
    
    Metrics:
    - MCD: mel cepstral distortion (lower = better)
    - Speaker Similarity: cosine sim of wav2vec embeddings (higher = better)
    """
    data_cfg = hps.data if isinstance(hps.data, dict) else hps.data.__dict__
    
    # Load test pairs
    with open(test_filelist) as f:
        lines = [l.strip().split('|') for l in f if l.strip()]
    
    if len(lines) < 2:
        return {'MCD': 0, 'SpkSim': 0, 'n_samples': 0}
    
    # Group by speaker
    spk_files = {}
    for parts in lines:
        wav_path = parts[0]
        spk = parts[1] if len(parts) > 1 else 'default'
        if spk not in spk_files:
            spk_files[spk] = []
        spk_files[spk].append(wav_path)
    
    speakers = list(spk_files.keys())
    if len(speakers) < 2:
        return {'MCD': 0, 'SpkSim': 0, 'n_samples': 0}
    
    # Load SSL model for speaker embedding extraction
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    ssl_model = bundle.get_model().to(device).eval()
    
    mcds = []
    spk_sims = []
    model.eval()
    
    n_samples = min(max_samples, len(lines))
    
    for i in tqdm(range(n_samples), desc="VC Quality"):
        # Pick source and target from different speakers
        src_spk = speakers[i % len(speakers)]
        tgt_spk = speakers[(i + 1) % len(speakers)]
        
        if not spk_files[src_spk] or not spk_files[tgt_spk]:
            continue
        
        src_path = spk_files[src_spk][i % len(spk_files[src_spk])]
        tgt_path = spk_files[tgt_spk][i % len(spk_files[tgt_spk])]
        
        try:
            # Extract source features
            src_feat = extract_ssl_features(src_path, device, 'wav2vec2')
            src_feat = src_feat.unsqueeze(0)  # (1, dim, T)
            
            # Extract F0
            src_wav, _ = torchaudio.load(src_path)
            f0 = extract_f0(src_wav).to(device)
            T = src_feat.size(-1)
            if f0.size(-1) > T:
                f0 = f0[:, :T]
            else:
                f0 = torch.nn.functional.pad(f0, (0, T - f0.size(-1)))
            f0 = f0.unsqueeze(0)
            
            # Load target audio for speaker embedding reference
            tgt_wav, tgt_sr = torchaudio.load(tgt_path)
            if tgt_sr != 16000:
                tgt_wav = torchaudio.functional.resample(tgt_wav, tgt_sr, 16000)
            if tgt_wav.size(0) > 1:
                tgt_wav = tgt_wav.mean(0, keepdim=True)
            
            tgt_mel = mel_spectrogram_torch(
                tgt_wav.to(device),
                data_cfg.get('filter_length', 1280) if isinstance(data_cfg, dict) else getattr(data_cfg, 'filter_length', 1280),
                data_cfg.get('n_mel_channels', 80) if isinstance(data_cfg, dict) else getattr(data_cfg, 'n_mel_channels', 80),
                16000,
                data_cfg.get('hop_length', 320) if isinstance(data_cfg, dict) else getattr(data_cfg, 'hop_length', 320),
                data_cfg.get('win_length', 1280) if isinstance(data_cfg, dict) else getattr(data_cfg, 'win_length', 1280),
                data_cfg.get('mel_fmin', 0.0) if isinstance(data_cfg, dict) else getattr(data_cfg, 'mel_fmin', 0.0),
                data_cfg.get('mel_fmax', None) if isinstance(data_cfg, dict) else getattr(data_cfg, 'mel_fmax', None)
            )
            
            # Convert — use speaker ID if use_spk=True
            spk_list = sorted(spk_files.keys())
            with torch.no_grad():
                c_lengths = torch.tensor([T], device=device)
                if hasattr(model, 'emb_g'):
                    tgt_spk_id = torch.LongTensor([spk_list.index(tgt_spk)]).to(device)
                    audio = model.infer(src_feat, g=tgt_spk_id, f0=f0, mel=None, c_lengths=c_lengths)
                else:
                    audio = model.infer(src_feat, g=None, f0=f0, mel=tgt_mel, c_lengths=c_lengths)
            
            # Compute converted mel for MCD
            conv_mel = mel_spectrogram_torch(
                audio.squeeze(1),
                data_cfg.get('filter_length', 1280) if isinstance(data_cfg, dict) else getattr(data_cfg, 'filter_length', 1280),
                data_cfg.get('n_mel_channels', 80) if isinstance(data_cfg, dict) else getattr(data_cfg, 'n_mel_channels', 80),
                16000,
                data_cfg.get('hop_length', 320) if isinstance(data_cfg, dict) else getattr(data_cfg, 'hop_length', 320),
                data_cfg.get('win_length', 1280) if isinstance(data_cfg, dict) else getattr(data_cfg, 'win_length', 1280),
                data_cfg.get('mel_fmin', 0.0) if isinstance(data_cfg, dict) else getattr(data_cfg, 'mel_fmin', 0.0),
                data_cfg.get('mel_fmax', None) if isinstance(data_cfg, dict) else getattr(data_cfg, 'mel_fmax', None)
            )
            
            # MCD with target
            mcd = mel_cepstral_distortion(tgt_mel, conv_mel)
            mcds.append(mcd)
            
            # Speaker similarity via wav2vec2 embeddings (converted vs target)
            conv_wav = audio.squeeze(0).cpu()
            tgt_spk_emb = get_speaker_embedding_from_wav(tgt_wav, 16000, ssl_model, device)
            conv_spk_emb = get_speaker_embedding_from_wav(conv_wav, 16000, ssl_model, device)
            sim = speaker_similarity_cosine(
                conv_spk_emb.unsqueeze(0), tgt_spk_emb.unsqueeze(0)
            )
            spk_sims.append(sim)
        
        except Exception as e:
            print(f"  Skip {i}: {e}")
            continue
    
    del ssl_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'MCD': np.mean(mcds) if mcds else 0,
        'SpkSim': np.mean(spk_sims) if spk_sims else 0,
        'n_samples': len(mcds),
    }


# ============================================================
# Metric 3: Noise Robustness
# ============================================================

def evaluate_noise_robustness(model, test_filelist, hps, device='cuda', 
                              noise_levels=[0, 0.01, 0.03, 0.05, 0.1],
                              max_samples=20):
    """
    Measure how VC quality degrades under input noise.
    CausalVoice should degrade LESS than baseline.
    """
    data_cfg = hps.data if isinstance(hps.data, dict) else hps.data.__dict__
    
    with open(test_filelist) as f:
        lines = [l.strip().split('|') for l in f if l.strip()][:max_samples]
    
    results = {nl: [] for nl in noise_levels}
    model.eval()
    
    for parts in tqdm(lines, desc="Noise Robustness"):
        wav_path = parts[0]
        try:
            # Load audio
            wav, sr = torchaudio.load(wav_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)
            
            # Clean reference
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            ssl_model = bundle.get_model().to(device).eval()
            
            with torch.no_grad():
                clean_feats, _ = ssl_model.extract_features(wav.to(device))
                clean_feat = clean_feats[-1].squeeze(0).T.unsqueeze(0)  # (1, dim, T)
            
            f0 = extract_f0(wav).to(device)
            T = clean_feat.size(-1)
            if f0.size(-1) != T:
                f0 = torch.nn.functional.pad(f0[:, :T], (0, max(0, T - f0.size(-1))))
            f0 = f0.unsqueeze(0)
            
            # Get clean output embedding for reference
            with torch.no_grad():
                clean_emb = model.get_causal_embeddings(clean_feat, c_lengths=torch.tensor([T], device=device))
            
            for noise_level in noise_levels:
                if noise_level == 0:
                    noisy_wav = wav
                else:
                    noisy_wav = wav + torch.randn_like(wav) * noise_level
                
                with torch.no_grad():
                    noisy_feats, _ = ssl_model.extract_features(noisy_wav.to(device))
                    noisy_feat = noisy_feats[-1].squeeze(0).T.unsqueeze(0)
                    noisy_emb = model.get_causal_embeddings(noisy_feat, c_lengths=torch.tensor([noisy_feat.size(-1)], device=device))
                
                # Measure embedding stability (how much does embedding change with noise)
                stability = torch.nn.functional.cosine_similarity(
                    clean_emb, noisy_emb, dim=-1
                ).item()
                results[noise_level].append(stability)
            
            del ssl_model  # Free memory
            
        except Exception as e:
            print(f"  Skip: {e}")
            continue
    
    return {nl: np.mean(vals) if vals else 0 for nl, vals in results.items()}


# ============================================================
# Metric 4: Cross-Domain Generalization
# ============================================================

def evaluate_cross_domain(model, train_filelist, test_filelist, hps, device='cuda', max_samples=30):
    """
    Evaluate generalization to unseen speakers.
    
    Compare MCD on seen vs unseen speakers.
    Lower MCD gap = better generalization.
    """
    data_cfg = hps.data if isinstance(hps.data, dict) else hps.data.__dict__
    
    # Get training speakers
    with open(train_filelist) as f:
        train_lines = [l.strip().split('|') for l in f if l.strip()]
    train_speakers = set(parts[1] for parts in train_lines if len(parts) > 1)
    
    # Get test speakers
    with open(test_filelist) as f:
        test_lines = [l.strip().split('|') for l in f if l.strip()]
    
    # Separate seen vs unseen
    seen_lines = [l for l in test_lines if len(l) > 1 and l[1] in train_speakers]
    unseen_lines = [l for l in test_lines if len(l) > 1 and l[1] not in train_speakers]
    
    # If no unseen speakers in val set, split train speakers into "seen" vs "held-out"
    if not unseen_lines:
        all_speakers = sorted(train_speakers)
        n_held = max(2, len(all_speakers) // 5)  # hold out 20%
        held_out_speakers = set(all_speakers[-n_held:])
        seen_lines = [l for l in test_lines if len(l) > 1 and l[1] not in held_out_speakers]
        unseen_lines = [l for l in test_lines if len(l) > 1 and l[1] in held_out_speakers]
    
    def compute_avg_mcd(lines, label):
        mcds = []
        model.eval()
        spk_files = {}
        for parts in lines:
            spk = parts[1] if len(parts) > 1 else 'default'
            if spk not in spk_files:
                spk_files[spk] = []
            spk_files[spk].append(parts[0])
        
        speakers = list(spk_files.keys())
        if len(speakers) < 2:
            return 0.0
        
        n = min(max_samples, len(lines))
        for i in range(n):
            src_spk = speakers[i % len(speakers)]
            tgt_spk = speakers[(i + 1) % len(speakers)]
            src_path = spk_files[src_spk][i % len(spk_files[src_spk])]
            tgt_path = spk_files[tgt_spk][i % len(spk_files[tgt_spk])]
            
            try:
                src_feat = extract_ssl_features(src_path, device, 'wav2vec2').unsqueeze(0)
                src_wav, _ = torchaudio.load(src_path)
                f0 = extract_f0(src_wav).to(device)
                T = src_feat.size(-1)
                if f0.size(-1) > T:
                    f0 = f0[:, :T]
                else:
                    f0 = torch.nn.functional.pad(f0, (0, T - f0.size(-1)))
                f0 = f0.unsqueeze(0)
                
                tgt_wav, tgt_sr = torchaudio.load(tgt_path)
                if tgt_sr != 16000:
                    tgt_wav = torchaudio.functional.resample(tgt_wav, tgt_sr, 16000)
                if tgt_wav.size(0) > 1:
                    tgt_wav = tgt_wav.mean(0, keepdim=True)
                
                filter_length = data_cfg.get('filter_length', 1280) if isinstance(data_cfg, dict) else getattr(data_cfg, 'filter_length', 1280)
                n_mel = data_cfg.get('n_mel_channels', 80) if isinstance(data_cfg, dict) else getattr(data_cfg, 'n_mel_channels', 80)
                hop = data_cfg.get('hop_length', 320) if isinstance(data_cfg, dict) else getattr(data_cfg, 'hop_length', 320)
                win = data_cfg.get('win_length', 1280) if isinstance(data_cfg, dict) else getattr(data_cfg, 'win_length', 1280)
                fmin = data_cfg.get('mel_fmin', 0.0) if isinstance(data_cfg, dict) else getattr(data_cfg, 'mel_fmin', 0.0)
                fmax = data_cfg.get('mel_fmax', None) if isinstance(data_cfg, dict) else getattr(data_cfg, 'mel_fmax', None)
                
                tgt_mel = mel_spectrogram_torch(tgt_wav.to(device), filter_length, n_mel, 16000, hop, win, fmin, fmax)
                
                spk_list = sorted(spk_files.keys())
                with torch.no_grad():
                    c_lengths = torch.tensor([T], device=device)
                    if hasattr(model, 'emb_g'):
                        tgt_spk_id = torch.LongTensor([spk_list.index(tgt_spk)]).to(device)
                        audio = model.infer(src_feat, g=tgt_spk_id, f0=f0, mel=None, c_lengths=c_lengths)
                    else:
                        audio = model.infer(src_feat, g=None, f0=f0, mel=tgt_mel, c_lengths=c_lengths)
                
                conv_mel = mel_spectrogram_torch(audio.squeeze(1), filter_length, n_mel, 16000, hop, win, fmin, fmax)
                mcds.append(mel_cepstral_distortion(tgt_mel, conv_mel))
            except Exception as e:
                continue
        
        return np.mean(mcds) if mcds else 0.0
    
    mcd_seen = compute_avg_mcd(seen_lines, "seen")
    mcd_unseen = compute_avg_mcd(unseen_lines, "unseen")
    generalization_gap = mcd_unseen - mcd_seen  # Lower = better generalization
    
    return {
        'MCD_seen': mcd_seen,
        'MCD_unseen': mcd_unseen,
        'generalization_gap': generalization_gap,
        'n_seen': len(seen_lines),
        'n_unseen': len(unseen_lines),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CausalVoice Full Evaluation")
    parser.add_argument('--baseline_ckpt', type=str, required=True)
    parser.add_argument('--causal_ckpt', type=str, required=True)
    parser.add_argument('--ablation_ckpts', type=str, default='',
                        help='Comma-separated paths for ablation checkpoints')
    parser.add_argument('--ablation_names', type=str, default='',
                        help='Comma-separated names for ablation models')
    parser.add_argument('--test_filelist', type=str, default='filelists/val_real.txt')
    parser.add_argument('--train_filelist', type=str, default='filelists/train_real.txt')
    parser.add_argument('--causal_test_path', type=str, default='data/causal_interventions')
    parser.add_argument('--config', type=str, default='configs/causal_vc.json')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_samples', type=int, default=50)
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hps = utils.get_hparams_from_file(args.config)
    
    # Build model list: baseline, causal, + ablation models
    model_configs = [
        ('baseline', args.baseline_ckpt),
        ('causal', args.causal_ckpt),
    ]
    if args.ablation_ckpts:
        abl_ckpts = args.ablation_ckpts.split(',')
        abl_names = args.ablation_names.split(',') if args.ablation_names else [f'ablation_{i}' for i in range(len(abl_ckpts))]
        for name, ckpt in zip(abl_names, abl_ckpts):
            if os.path.exists(ckpt.strip()):
                model_configs.append((name.strip(), ckpt.strip()))
    
    results = {}
    
    for name, ckpt_path in model_configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name.upper()} ({ckpt_path})")
        print(f"{'='*60}")
        
        model, _ = load_model(ckpt_path, args.config, device)
        
        # 1. Causal Awareness
        print("\n[1/4] Causal Awareness Metrics...")
        if os.path.exists(args.causal_test_path):
            causal_metrics = evaluate_causal_awareness(model, args.causal_test_path, hps, device)
        else:
            causal_metrics = {'HNC': 0, 'ARS': 0, 'DR': 0}
        print(f"  HNC={causal_metrics['HNC']:.4f}  ARS={causal_metrics['ARS']:.4f}  DR={causal_metrics['DR']:.4f}")
        
        # 2. VC Quality
        print("\n[2/4] VC Quality Metrics...")
        if os.path.exists(args.test_filelist):
            vc_metrics = evaluate_vc_quality(model, args.test_filelist, hps, device, args.max_samples)
        else:
            vc_metrics = {'MCD': 0, 'SpkSim': 0}
        print(f"  MCD={vc_metrics['MCD']:.4f}  SpkSim={vc_metrics['SpkSim']:.4f}")
        
        # 3. Noise Robustness
        print("\n[3/4] Noise Robustness...")
        if os.path.exists(args.test_filelist):
            robustness = evaluate_noise_robustness(model, args.test_filelist, hps, device, max_samples=20)
        else:
            robustness = {}
        for nl, val in robustness.items():
            print(f"  Noise={nl:.3f}: Stability={val:.4f}")
        
        # 4. Cross-Domain Generalization
        print("\n[4/4] Cross-Domain Generalization...")
        if os.path.exists(args.test_filelist) and os.path.exists(args.train_filelist):
            cross_domain = evaluate_cross_domain(model, args.train_filelist, args.test_filelist, hps, device, max_samples=30)
            print(f"  MCD_seen={cross_domain['MCD_seen']:.4f}  MCD_unseen={cross_domain['MCD_unseen']:.4f}  Gap={cross_domain['generalization_gap']:.4f}")
        else:
            cross_domain = {'MCD_seen': 0, 'MCD_unseen': 0, 'generalization_gap': 0}
        
        results[name] = {
            'causal': causal_metrics,
            'vc_quality': vc_metrics,
            'robustness': robustness,
            'cross_domain': cross_domain,
        }
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ============ Comparison Table ============
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    
    # Header
    model_names = list(results.keys())
    header = f"{'Metric':<20}" + "".join(f"{n:<15}" for n in model_names) + "Better?"
    print(f"\n{header}")
    print("-" * (20 + 15 * len(model_names) + 8))
    
    rows = [
        ('HNC ↑', 'causal', 'HNC', True),
        ('ARS ↑', 'causal', 'ARS', True),
        ('DR ↑', 'causal', 'DR', True),
        ('MCD ↓', 'vc_quality', 'MCD', False),
        ('SpkSim ↑', 'vc_quality', 'SpkSim', True),
        ('Gen.Gap ↓', 'cross_domain', 'generalization_gap', False),
    ]
    
    for label, category, metric, higher_better in rows:
        vals = [results[n][category].get(metric, 0) for n in model_names]
        best_idx = vals.index(max(vals)) if higher_better else vals.index(min(vals))
        line = f"{label:<20}"
        for i, v in enumerate(vals):
            mark = "★" if i == best_idx and len(model_names) > 1 else " "
            line += f"{v:<14.4f}{mark}"
        print(line)
    
    # Robustness comparison
    if any(results[n]['robustness'] for n in model_names):
        print(f"\n{'Noise Level':<12}", end="")
        for n in model_names:
            print(f"{n:<15}", end="")
        print()
        print("-" * (12 + 15 * len(model_names)))
        noise_levels = sorted(results[model_names[0]]['robustness'].keys())
        for nl in noise_levels:
            line = f"{nl:<12.3f}"
            for n in model_names:
                v = results[n]['robustness'].get(nl, 0)
                line += f"{v:<15.4f}"
            print(line)
    
    # Ablation summary
    if len(model_names) > 2:
        print(f"\n{'='*70}")
        print("ABLATION STUDY")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'ARS':<10} {'DR':<10} {'MCD':<10} {'SpkSim':<10} {'Gen.Gap':<10}")
        print("-" * 70)
        for n in model_names:
            r = results[n]
            print(f"{n:<20} {r['causal']['ARS']:<10.4f} {r['causal']['DR']:<10.4f} "
                  f"{r['vc_quality']['MCD']:<10.4f} {r['vc_quality']['SpkSim']:<10.4f} "
                  f"{r['cross_domain']['generalization_gap']:<10.4f}")
    
    # Save results
    save_path = output_dir / "comparison_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
