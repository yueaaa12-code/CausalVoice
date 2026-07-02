"""
CausalVoice Feasibility Verification Script.

Generates synthetic causal intervention data (no real audio needed),
runs a short training loop on the causal projector alone, and verifies
that causal regularization produces meaningful embedding structure.

Metrics:
- HNC (Hit at Non-Causal): embeddings of non-causal pairs should be similar
- ARS (Average Ranking Score): higher causal effect → larger embedding distance
- Discrimination Ratio: high-CE distance / low-CE distance
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import json
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local imports
from models_causal import SynthesizerTrnCausal
from causal.projector import CausalProjector
from causal.losses import calc_contrastive_loss, calc_ranking_loss


def compute_causal_metrics(embeds, causal_effects, data_splits, non_causal_thresh=0.05, causal_thresh=0.1):
    """
    Compute causal awareness metrics:
    - HNC: fraction of non-causal interventions where model incorrectly shows high sensitivity
    - ARS: average ranking score (Kendall's tau between CE and distance)
    - Discrimination ratio: mean distance for high-CE / mean distance for low-CE
    """
    hnc_total = 0
    hnc_count = 0
    high_ce_dists = []
    low_ce_dists = []
    rank_correct = 0
    rank_total = 0
    
    with torch.no_grad():
        for sample_id in range(len(causal_effects)):
            anchor = embeds[data_splits[sample_id]]
            keys = embeds[data_splits[sample_id] + 1:data_splits[sample_id + 1]]
            ce = causal_effects[sample_id].cpu().numpy()
            
            # Cosine distances
            dists = (1.0 - torch.matmul(keys, anchor)).cpu().numpy()
            
            # HNC: non-causal items with high sensitivity
            non_causal_mask = ce < non_causal_thresh
            if non_causal_mask.any():
                hnc_total += (dists[non_causal_mask] > causal_thresh).sum()
                hnc_count += non_causal_mask.sum()
                low_ce_dists.extend(dists[non_causal_mask].tolist())
            
            # High CE items
            high_ce_mask = ce > 0.3
            if high_ce_mask.any():
                high_ce_dists.extend(dists[high_ce_mask].tolist())
            
            # Pairwise ranking correctness
            for i in range(len(ce)):
                for j in range(i+1, len(ce)):
                    if abs(ce[i] - ce[j]) > 0.1:
                        rank_total += 1
                        if (ce[i] > ce[j]) == (dists[i] > dists[j]):
                            rank_correct += 1
    
    hnc = 1.0 - (hnc_total / max(hnc_count, 1))  # 1 = perfect (no hallucinations)
    ars = rank_correct / max(rank_total, 1)  # 1 = perfect ranking
    
    mean_high = np.mean(high_ce_dists) if high_ce_dists else 0.0
    mean_low = np.mean(low_ce_dists) if low_ce_dists else 1e-6
    disc_ratio = mean_high / max(mean_low, 1e-6)
    
    return {'hnc': hnc, 'ars': ars, 'discrimination_ratio': disc_ratio,
            'mean_high_ce_dist': mean_high, 'mean_low_ce_dist': mean_low}


def generate_synthetic_groups(num_groups=100, group_size=5, ssl_dim=1024, 
                               seq_len=50, device='cpu'):
    """
    Generate synthetic causal intervention groups with CONFOUNDERS.
    
    Key insight: We need features where magnitude of change ≠ causal effect.
    This mirrors real speech where:
    - Background noise change = LARGE acoustic shift but LOW causal effect on content
    - Subtle speaker identity = SMALL acoustic shift but HIGH causal effect on timbre
    
    Without this confounding, the projector has nothing to learn (baseline already perfect).
    """
    groups = []
    for g in range(num_groups):
        # Random "anchor" WavLM-like features
        anchor = torch.randn(ssl_dim, seq_len, device=device) * 0.5
        
        items = [{'features': anchor.clone(), 'causal_effect': 0.0}]
        
        for i in range(group_size - 1):
            intervention_type = np.random.choice(
                ['speaker_subtle', 'noise_large', 'speed_medium', 'room_large'],
                p=[0.25, 0.3, 0.2, 0.25]
            )
            
            if intervention_type == 'speaker_subtle':
                # Speaker change: SMALL feature shift but HIGH causal effect
                # (speaker identity is encoded in subtle patterns, not magnitude)
                direction = torch.randn(ssl_dim, 1, device=device)
                direction = direction / direction.norm() * 0.3  # small magnitude
                shift = direction.expand_as(anchor) + torch.randn_like(anchor) * 0.05
                features = anchor + shift
                ce = np.random.uniform(0.5, 1.0)  # HIGH causal effect
                
            elif intervention_type == 'noise_large':
                # Noise/reverb: LARGE feature shift but LOW causal effect
                # (additive noise/reverb changes features a lot but shouldn't affect content)
                noise = torch.randn_like(anchor) * 1.5  # large magnitude
                features = anchor + noise
                ce = np.random.uniform(0.0, 0.04)  # LOW causal effect
                
            elif intervention_type == 'speed_medium':
                # Speed change: medium feature shift, medium causal effect
                shift = torch.randn_like(anchor) * 0.6
                features = anchor + shift
                ce = np.random.uniform(0.15, 0.45)
                
            else:  # room_large
                # Room acoustics: LARGE structured shift but LOW causal effect
                # (reverb adds consistent pattern across time)
                reverb_pattern = torch.randn(ssl_dim, 1, device=device) * 1.2
                features = anchor + reverb_pattern.expand_as(anchor) + torch.randn_like(anchor) * 0.3
                ce = np.random.uniform(0.0, 0.05)  # LOW causal effect
            
            items.append({'features': features, 'causal_effect': ce})
        
        groups.append(items)
    
    return groups


def groups_to_batch(groups, device='cpu'):
    """Convert groups to flat batch with data_splits (CausalSim2Real pattern)."""
    all_features = []
    all_causal_effects = []
    data_splits = [0]
    
    for group in groups:
        group_ces = []
        for item in group:
            all_features.append(item['features'])
            group_ces.append(item['causal_effect'])
        # CE list excludes anchor (first item)
        all_causal_effects.append(torch.tensor(group_ces[1:], dtype=torch.float32, device=device))
        data_splits.append(data_splits[-1] + len(group))
    
    features = torch.stack(all_features)  # (N, ssl_dim, T)
    return features, all_causal_effects, data_splits


def run_feasibility_test(config=None):
    """Main verification: train causal projector and check metrics."""
    
    if config is None:
        config = {
            'num_groups': 500,
            'group_size': 5,
            'ssl_dim': 1024,
            'seq_len': 50,
            'hidden_channels': 192,
            'proj_dim': 32,
            'batch_groups': 32,
            'num_steps': 2000,
            'lr': 1e-3,
            'lambda_contrastive': 1.0,
            'lambda_ranking': 5.0,
            'tau': 0.07,
            'non_causal_thresh': 0.05,
            'causal_thresh': 0.1,
            'ranking_margin': 0.2,
            'unfreeze_enc_p': True,
            'weight_decay': 1e-4,
            'lr_schedule': 'cosine',  # cosine annealing
            'early_stop_patience': 300,  # steps without improvement
        }
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print("=" * 60)
    
    # ============ Step 1: Create model (only enc_p + projector) ============
    print("\n[Step 1] Creating model...")
    model = SynthesizerTrnCausal(
        spec_channels=513, segment_size=32,
        inter_channels=192, hidden_channels=config['hidden_channels'],
        filter_channels=768, n_heads=2, n_layers=6,
        kernel_size=3, p_dropout=0.0,
        resblock='1', resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 8, 2, 2],
        upsample_initial_channel=512, upsample_kernel_sizes=[16, 16, 4, 4],
        gin_channels=256, ssl_dim=config['ssl_dim'], use_spk=False
    ).to(device)
    
    # Only train the causal projector (enc_p is frozen anyway)
    # But we also want to verify that enc_p hidden states respond to different inputs
    wd = config.get('weight_decay', 0)
    if config.get('unfreeze_enc_p', False):
        # Unfreeze enc_p for causal learning (key insight: projector alone can't overcome confounders)
        for param in model.enc_p.parameters():
            param.requires_grad = True
        trainable_params = list(model.enc_p.parameters()) + list(model.causal_projector.parameters())
        optimizer = AdamW(trainable_params, lr=config['lr'], weight_decay=wd)
        print(f"  Training enc_p + projector (unfrozen)")
    else:
        optimizer = AdamW(model.causal_projector.parameters(), lr=config['lr'], weight_decay=wd)
        print(f"  Training projector only (enc_p frozen)")
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_steps'], eta_min=config['lr'] * 0.01)
    
    total_params = sum(p.numel() for p in model.causal_projector.parameters())
    print(f"  Causal projector params: {total_params:,}")
    
    # ============ Step 2: Generate synthetic data ============
    print("\n[Step 2] Generating synthetic causal intervention data...")
    t0 = time.time()
    groups = generate_synthetic_groups(
        num_groups=config['num_groups'],
        group_size=config['group_size'],
        ssl_dim=config['ssl_dim'],
        seq_len=config['seq_len'],
        device=device
    )
    print(f"  Generated {len(groups)} groups ({len(groups) * config['group_size']} samples) in {time.time()-t0:.2f}s")
    
    # ============ Step 3: Baseline metrics (before training) ============
    print("\n[Step 3] Computing baseline metrics (before training)...")
    model.eval()
    with torch.no_grad():
        eval_groups = groups[:32]
        features, causal_effects, data_splits = groups_to_batch(eval_groups, device)
        c_lengths = torch.full((features.size(0),), features.size(2), device=device).long()
        embeds = model.get_causal_embeddings(features, c_lengths=c_lengths)
        
        metrics_before = compute_causal_metrics(embeds, causal_effects, data_splits)
    
    print(f"  HNC (before):  {metrics_before['hnc']:.4f}  (target: → 1.0)")
    print(f"  ARS (before):  {metrics_before['ars']:.4f}  (target: → 1.0)")
    print(f"  Disc Ratio:    {metrics_before['discrimination_ratio']:.4f}  (target: >> 1.0)")
    
    # ============ Step 4: Training loop ============
    print(f"\n[Step 4] Training for {config['num_steps']} steps (with early stopping)...")
    model.train()
    
    losses_history = []
    bg = config['batch_groups']
    best_score = -float('inf')
    best_step = 0
    best_metrics = None
    best_state = None
    
    # Eval groups for metric tracking
    eval_groups = groups[:32]
    
    for step in range(config['num_steps']):
        # Sample a mini-batch of groups
        idx = np.random.choice(len(groups), bg, replace=False)
        batch_groups = [groups[i] for i in idx]
        
        features, causal_effects, data_splits = groups_to_batch(batch_groups, device)
        c_lengths = torch.full((features.size(0),), features.size(2), device=device).long()
        
        # Forward
        embeds = model.get_causal_embeddings(features, c_lengths=c_lengths)
        
        # Compute causal losses
        loss_c = calc_contrastive_loss(
            embeds, causal_effects, data_splits,
            contrastive_weight=config['lambda_contrastive'],
            tau=config['tau'],
            non_causal_thresh=config['non_causal_thresh'],
            causal_thresh=config['causal_thresh']
        )
        loss_r = calc_ranking_loss(
            embeds, causal_effects, data_splits,
            ranking_weight=config['lambda_ranking'],
            margin=config['ranking_margin']
        )
        
        loss = loss_c + loss_r
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses_history.append({
            'total': loss.item(),
            'contrastive': loss_c.item(),
            'ranking': loss_r.item()
        })
        
        if (step + 1) % 50 == 0:
            avg_loss = np.mean([l['total'] for l in losses_history[-50:]])
            avg_c = np.mean([l['contrastive'] for l in losses_history[-50:]])
            avg_r = np.mean([l['ranking'] for l in losses_history[-50:]])
            print(f"  Step {step+1:4d} | Loss: {avg_loss:.4f} (C: {avg_c:.4f}, R: {avg_r:.4f}) lr={scheduler.get_last_lr()[0]:.6f}")
        
        # Evaluate and early stop every 200 steps
        if (step + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                ev_feat, ev_ce, ev_ds = groups_to_batch(eval_groups, device)
                ev_len = torch.full((ev_feat.size(0),), ev_feat.size(2), device=device).long()
                ev_emb = model.get_causal_embeddings(ev_feat, c_lengths=ev_len)
                mid_metrics = compute_causal_metrics(ev_emb, ev_ce, ev_ds)
            
            # Combined score: HNC + ARS + min(DR/10, 1)
            score = mid_metrics['hnc'] + mid_metrics['ars'] + min(mid_metrics['discrimination_ratio'] / 10.0, 1.0)
            print(f"    → Metrics@{step+1}: HNC={mid_metrics['hnc']:.3f} ARS={mid_metrics['ars']:.3f} DR={mid_metrics['discrimination_ratio']:.2f} Score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_step = step + 1
                best_metrics = mid_metrics.copy()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif (step + 1) - best_step > config.get('early_stop_patience', 500):
                print(f"    ★ Early stopping at step {step+1} (best was step {best_step})")
                break
            model.train()
    
    # ============ Step 5: Post-training metrics (using best checkpoint) ============
    print(f"\n[Step 5] Loading best checkpoint (step {best_step})...")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        features, causal_effects, data_splits = groups_to_batch(eval_groups, device)
        c_lengths = torch.full((features.size(0),), features.size(2), device=device).long()
        embeds = model.get_causal_embeddings(features, c_lengths=c_lengths)
        
        metrics_after = compute_causal_metrics(embeds, causal_effects, data_splits)
    
    print(f"  HNC (after):   {metrics_after['hnc']:.4f}  (was {metrics_before['hnc']:.4f})")
    print(f"  ARS (after):   {metrics_after['ars']:.4f}  (was {metrics_before['ars']:.4f})")
    print(f"  Disc Ratio:    {metrics_after['discrimination_ratio']:.4f}  (was {metrics_before['discrimination_ratio']:.4f})")
    
    # ============ Step 6: Generalization test on UNSEEN data ============
    print("\n[Step 6] Generalization test on unseen data...")
    test_groups = generate_synthetic_groups(
        num_groups=100, group_size=config['group_size'],
        ssl_dim=config['ssl_dim'], seq_len=config['seq_len'], device=device
    )
    with torch.no_grad():
        test_features, test_ce, test_ds = groups_to_batch(test_groups, device)
        test_len = torch.full((test_features.size(0),), test_features.size(2), device=device).long()
        test_embeds = model.get_causal_embeddings(test_features, c_lengths=test_len)
        test_metrics = compute_causal_metrics(test_embeds, test_ce, test_ds)
    print(f"  HNC (test):    {test_metrics['hnc']:.4f}")
    print(f"  ARS (test):    {test_metrics['ars']:.4f}")
    print(f"  Disc Ratio:    {test_metrics['discrimination_ratio']:.4f}")
    print(f"  Mean dist (high-CE): {test_metrics['mean_high_ce_dist']:.4f}")
    print(f"  Mean dist (low-CE):  {test_metrics['mean_low_ce_dist']:.4f}")
    
    # ============ Step 7: Verdict ============
    print("\n" + "=" * 60)
    print("FEASIBILITY VERDICT:")
    print("=" * 60)
    
    hnc_maintained = metrics_after['hnc'] >= 0.9  # HNC should stay high (not degrade)
    ars_improved = metrics_after['ars'] > metrics_before['ars'] + 0.05
    disc_improved = metrics_after['discrimination_ratio'] > 2.0  # Should be >> 1.0
    generalizes = test_metrics['discrimination_ratio'] > 2.0  # Works on unseen data
    
    results = {
        'hnc_maintained': hnc_maintained,
        'ars_improved': ars_improved,
        'disc_improved': disc_improved,
        'generalizes': generalizes,
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'test_metrics': test_metrics,
        'final_loss': losses_history[-1]['total'],
        'initial_loss': losses_history[0]['total'],
        'best_step': best_step,
    }
    
    checks = [
        (f"HNC maintained ≥0.9: {metrics_after['hnc']:.4f}", hnc_maintained),
        (f"ARS improved: {metrics_before['ars']:.4f} → {metrics_after['ars']:.4f}", ars_improved),
        (f"Disc ratio > 2.0: {metrics_after['discrimination_ratio']:.4f}", disc_improved),
        (f"Generalizes to unseen: DR={test_metrics['discrimination_ratio']:.4f}", generalizes),
    ]
    
    all_pass = True
    for desc, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {desc}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("  ★ FEASIBILITY CONFIRMED: Causal regularization produces meaningful embeddings.")
    elif sum(1 for _, p in checks if p) >= 2:
        print("  △ PARTIAL SUCCESS: Some metrics improved. May need hyperparameter tuning.")
    else:
        print("  ✗ NEEDS INVESTIGATION: Causal losses not producing expected structure.")
    
    return results


if __name__ == '__main__':
    results = run_feasibility_test()
