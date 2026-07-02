"""
Causal awareness evaluation for CausalVoice.

Evaluates how well the model distinguishes causal from non-causal factors
using HNC (Hallucinated Non-Causal) and ARS (Average Response Sensitivity) metrics.

Adapted from CausalSim2Real's HNC_ARS evaluation.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from models_causal import SynthesizerTrnCausal
from causal import (
    CausalInterventionDataset,
    causal_collate_fn,
    calc_causal_awareness_metrics
)
import utils
import json


def evaluate_causal(args):
    """
    Evaluate causal awareness of the model.
    
    Metrics:
    - HNC (Hallucinated Non-Causal): how many non-causal interventions 
      the model incorrectly responds to (lower is better)
    - ARS (Average Response Sensitivity): average embedding distance 
      for non-causal interventions (lower is better)
    - Causal Sensitivity: average embedding distance for truly causal 
      interventions (higher is better for discrimination)
    """
    # Load config
    model_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(model_dir, '..', 'configs', 'causal_vc.json')
    if not os.path.exists(config_path):
        config_path = args.config
    
    with open(config_path) as f:
        hps = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_cfg = hps['model']
    net_g = SynthesizerTrnCausal(
        hps['data']['filter_length'] // 2 + 1,
        hps['train']['segment_size'] // hps['data']['hop_length'],
        **model_cfg
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        net_g.load_state_dict(checkpoint['model'], strict=False)
    else:
        net_g.load_state_dict(checkpoint, strict=False)
    net_g.eval()
    
    # Load causal test data
    causal_dataset = CausalInterventionDataset(args.causal_test_path, 
                                                argparse.Namespace(**hps))
    causal_loader = DataLoader(
        causal_dataset, batch_size=16, shuffle=False,
        num_workers=4, collate_fn=causal_collate_fn
    )
    
    # Evaluate
    total_hnc = 0
    all_ars = []
    all_causal_sensitivity = []
    total_non_causal = 0
    total_causal = 0
    
    with torch.no_grad():
        for batch_idx, (features, spk, f0, causal_effects, data_splits) in enumerate(causal_loader):
            features = features.to(device)
            spk = spk.to(device)
            causal_effects_gpu = [ce.to(device) for ce in causal_effects]
            
            # Get causal embeddings
            embeds = net_g.get_causal_embeddings(features, spk)
            
            # Compute metrics
            hnc, ars = calc_causal_awareness_metrics(
                embeds, causal_effects_gpu, data_splits,
                non_causal_thresh=args.non_causal_thresh,
                causal_thresh=args.causal_thresh
            )
            total_hnc += hnc
            all_ars.extend(ars)
            
            # Also compute causal sensitivity
            for sample_id in range(len(causal_effects)):
                q = embeds[data_splits[sample_id]]
                keys = embeds[data_splits[sample_id] + 1:data_splits[sample_id + 1]]
                ce = causal_effects[sample_id].numpy()
                
                sensitivities = (1.0 - torch.matmul(keys, q)).cpu().numpy()
                
                causal_mask = ce >= args.causal_thresh
                non_causal_mask = ce < args.non_causal_thresh
                
                total_non_causal += non_causal_mask.sum()
                total_causal += causal_mask.sum()
                
                if causal_mask.any():
                    all_causal_sensitivity.extend(sensitivities[causal_mask].tolist())
    
    # Compute final metrics
    avg_ars = np.mean(all_ars) if all_ars else 0.0
    avg_causal_sens = np.mean(all_causal_sensitivity) if all_causal_sensitivity else 0.0
    hnc_rate = total_hnc / max(total_non_causal, 1)
    
    # Discrimination ratio: causal sensitivity / non-causal sensitivity
    discrimination = avg_causal_sens / max(avg_ars, 1e-8)
    
    print("=" * 60)
    print("CausalVoice - Causal Awareness Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data:  {args.causal_test_path}")
    print("-" * 60)
    print(f"HNC (Hallucinated Non-Causal): {total_hnc}")
    print(f"HNC Rate:                      {hnc_rate:.4f}")
    print(f"ARS (Avg Response Sensitivity): {avg_ars:.6f}")
    print(f"Causal Sensitivity:            {avg_causal_sens:.6f}")
    print(f"Discrimination Ratio:          {discrimination:.2f}x")
    print("-" * 60)
    print(f"Total non-causal interventions: {total_non_causal}")
    print(f"Total causal interventions:     {total_causal}")
    print("=" * 60)
    
    # Save results
    results = {
        'hnc': int(total_hnc),
        'hnc_rate': float(hnc_rate),
        'ars': float(avg_ars),
        'causal_sensitivity': float(avg_causal_sens),
        'discrimination_ratio': float(discrimination),
        'total_non_causal': int(total_non_causal),
        'total_causal': int(total_causal),
    }
    
    output_path = os.path.join(os.path.dirname(args.checkpoint), 'causal_eval_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CausalVoice causal awareness")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument('--causal_test_path', type=str, required=True,
                        help="Path to causal intervention test data")
    parser.add_argument('--config', type=str, default="configs/causal_vc.json",
                        help="Config file path")
    parser.add_argument('--non_causal_thresh', type=float, default=0.02)
    parser.add_argument('--causal_thresh', type=float, default=0.1)
    args = parser.parse_args()
    
    evaluate_causal(args)
