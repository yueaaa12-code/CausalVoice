"""
Full evaluation pipeline for CausalVoice vs Baseline comparison.

Produces all metrics needed for paper:
1. Causal awareness: HNC, ARS, DR
2. VC quality: Speaker Similarity, MCD
3. Noise robustness: quality degradation under noise

Usage:
    python evaluate_pipeline.py \
        --baseline_ckpt logs/baseline/G_best.pth \
        --causal_ckpt logs/causal_voice/G_best.pth \
        --test_filelist filelists/val_real.txt \
        --causal_test_path data/synthetic_causal/test \
        --config configs/causal_vc.json \
        --output_dir results/
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
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).item()


def evaluate_vc_quality(model, test_filelist, hps, device='cuda', max_samples=50):
    """
    Evaluate VC quality: convert source → target speaker, measure quality.
    
    Metrics:
    - MCD: mel cepstral distortion (lower = better)
    - Speaker Similarity: cosine sim of speaker embeddings (higher = better)
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
            
            # Load target mel for speaker encoder
            tgt_wav, tgt_sr = torchaudio.load(tgt_path)
            if tgt_sr != 16000:
                tgt_wav = torchaudio.functional.resample(tgt_wav, tgt_sr, 16000)
            tgt_mel = mel_spectrogram_torch(
                tgt_wav.to(device),
                data_cfg.get('filter_length', 1280),
                data_cfg.get('n_mel_channels', 80),
                16000,
                data_cfg.get('hop_length', 320),
                data_cfg.get('win_length', 1280),
                data_cfg.get('mel_fmin', 0.0),
                data_cfg.get('mel_fmax', None)
            )
            
            # Convert — use speaker ID if use_spk=True
            with torch.no_grad():
                c_lengths = torch.tensor([T], device=device)
                if hasattr(model, 'emb_g'):
                    # use_spk=True: pass target speaker ID
                    # We use speaker index from the test filelist
                    spk_list = sorted(spk_files.keys())
                    tgt_spk_id = torch.LongTensor([spk_list.index(tgt_spk)]).to(device)
                    audio = model.infer(src_feat, g=tgt_spk_id, f0=f0, mel=None, c_lengths=c_lengths)
                else:
                    # use_spk=False: use mel-based speaker encoder
                    audio = model.infer(src_feat, g=None, f0=f0, mel=tgt_mel, c_lengths=c_lengths)
            
            # Compute converted mel
            conv_mel = mel_spectrogram_torch(
                audio.squeeze(1),
                data_cfg.get('filter_length', 1280),
                data_cfg.get('n_mel_channels', 80),
                16000,
                data_cfg.get('hop_length', 320),
                data_cfg.get('win_length', 1280),
                data_cfg.get('mel_fmin', 0.0),
                data_cfg.get('mel_fmax', None)
            )
            
            # MCD with target
            mcd = mel_cepstral_distortion(tgt_mel, conv_mel)
            mcds.append(mcd)
            
            # Speaker similarity via model's speaker encoder
            if hasattr(model, 'enc_spk'):
                with torch.no_grad():
                    spk_emb_conv = model.enc_spk.embed_utterance(conv_mel.squeeze(0).transpose(0, 1).unsqueeze(0))
                    spk_emb_tgt = model.enc_spk.embed_utterance(tgt_mel.squeeze(0).transpose(0, 1).unsqueeze(0))
                sim = speaker_similarity_cosine(spk_emb_conv, spk_emb_tgt)
                spk_sims.append(sim)
        
        except Exception as e:
            print(f"  Skip {i}: {e}")
            continue
    
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CausalVoice Full Evaluation")
    parser.add_argument('--baseline_ckpt', type=str, required=True)
    parser.add_argument('--causal_ckpt', type=str, required=True)
    parser.add_argument('--test_filelist', type=str, default='filelists/val_real.txt')
    parser.add_argument('--causal_test_path', type=str, default='data/synthetic_causal/test')
    parser.add_argument('--config', type=str, default='configs/causal_vc.json')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_samples', type=int, default=50)
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hps = utils.get_hparams_from_file(args.config)
    
    results = {}
    
    for name, ckpt_path in [('baseline', args.baseline_ckpt), ('causal', args.causal_ckpt)]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name.upper()} ({ckpt_path})")
        print(f"{'='*60}")
        
        model, _ = load_model(ckpt_path, args.config, device)
        
        # 1. Causal Awareness
        print("\n[1/3] Causal Awareness Metrics...")
        if os.path.exists(args.causal_test_path):
            causal_metrics = evaluate_causal_awareness(model, args.causal_test_path, hps, device)
        else:
            causal_metrics = {'HNC': 0, 'ARS': 0, 'DR': 0}
        print(f"  HNC={causal_metrics['HNC']:.4f}  ARS={causal_metrics['ARS']:.4f}  DR={causal_metrics['DR']:.4f}")
        
        # 2. VC Quality
        print("\n[2/3] VC Quality Metrics...")
        if os.path.exists(args.test_filelist):
            vc_metrics = evaluate_vc_quality(model, args.test_filelist, hps, device, args.max_samples)
        else:
            vc_metrics = {'MCD': 0, 'SpkSim': 0}
        print(f"  MCD={vc_metrics['MCD']:.4f}  SpkSim={vc_metrics['SpkSim']:.4f}")
        
        # 3. Noise Robustness
        print("\n[3/3] Noise Robustness...")
        if os.path.exists(args.test_filelist):
            robustness = evaluate_noise_robustness(model, args.test_filelist, hps, device, max_samples=20)
        else:
            robustness = {}
        for nl, val in robustness.items():
            print(f"  Noise={nl:.3f}: Stability={val:.4f}")
        
        results[name] = {
            'causal': causal_metrics,
            'vc_quality': vc_metrics,
            'robustness': robustness,
        }
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ============ Comparison Table ============
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    
    b, c = results['baseline'], results['causal']
    
    print(f"\n{'Metric':<25} {'Baseline':<12} {'CausalVoice':<12} {'Better?'}")
    print("-" * 60)
    
    rows = [
        ('HNC ↑', b['causal']['HNC'], c['causal']['HNC'], True),
        ('ARS ↑', b['causal']['ARS'], c['causal']['ARS'], True),
        ('DR ↑', b['causal']['DR'], c['causal']['DR'], True),
        ('MCD ↓', b['vc_quality']['MCD'], c['vc_quality']['MCD'], False),
        ('SpkSim ↑', b['vc_quality']['SpkSim'], c['vc_quality']['SpkSim'], True),
    ]
    
    for name, bv, cv, higher_better in rows:
        better = cv > bv if higher_better else cv < bv
        sym = '✓' if better else '✗'
        print(f"{name:<25} {bv:<12.4f} {cv:<12.4f} {sym}")
    
    # Robustness comparison
    if b['robustness'] and c['robustness']:
        print(f"\n{'Noise Level':<15} {'Baseline Stab':<15} {'Causal Stab':<15} {'Δ'}")
        print("-" * 55)
        for nl in sorted(b['robustness'].keys()):
            bv = b['robustness'].get(nl, 0)
            cv = c['robustness'].get(nl, 0)
            delta = cv - bv
            print(f"{nl:<15.3f} {bv:<15.4f} {cv:<15.4f} {delta:+.4f}")
    
    # Save results
    save_path = output_dir / "comparison_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
