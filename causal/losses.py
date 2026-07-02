"""
Causal regularization losses for CausalVoice.
Adapted from CausalSim2Real (CVPR 2025) for voice conversion embeddings.

Core idea: In synthetic data, we can intervene on factors (speaker, prosody, noise)
and measure causal effects. We use these to regularize the learned representations:
- Contrastive loss: non-causal interventions → embeddings should stay close
- Ranking loss: embeddings should be ordered by causal effect magnitude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def calc_contrastive_loss(embeds, causal_effects, data_splits,
                          contrastive_weight=1000.0, tau=0.2,
                          non_causal_thresh=0.02, causal_thresh=0.1):
    """
    InfoNCE-style contrastive loss based on causal effects.
    
    For each factual sample, interventions with low causal effect (non-causal)
    are positive pairs, and those with high causal effect are negative pairs.
    
    Args:
        embeds: (N_total,) tensor of L2-normalized embeddings from the causal projector
        causal_effects: list of tensors, each (n_interventions,) measuring how much
                       each intervention changed the output
        data_splits: list of indices marking where each factual sample's group starts
        contrastive_weight: scaling factor for the loss
        tau: temperature for softmax
        non_causal_thresh: causal effect below this → positive sample
        causal_thresh: causal effect above this → negative sample
    
    Returns:
        contrastive_loss: scalar tensor
    """
    infonces = []
    for sample_id in range(len(causal_effects)):
        # Factual embedding (the original synthetic utterance)
        q = embeds[data_splits[sample_id]]
        
        ce = causal_effects[sample_id]
        
        # Positives: interventions that barely changed the output (non-causal factors)
        positives = torch.where(ce <= non_causal_thresh)[0]
        # Negatives: interventions that significantly changed the output (causal factors)
        negatives = torch.where(ce >= causal_thresh)[0]
        
        if len(positives) > 0 and len(negatives) > 0:
            # Random positive
            pos_idx = positives[torch.randint(len(positives), (1,)).item()]
            k_plus = embeds[data_splits[sample_id] + 1 + pos_idx]
            
            # All negatives
            k_negs = embeds[data_splits[sample_id] + 1 + negatives]
            
            # InfoNCE
            numerator = torch.dot(q, k_plus) / tau
            denominator = torch.exp(numerator) + torch.sum(
                torch.exp(torch.matmul(k_negs, q) / tau)
            )
            infonces.append(-numerator + torch.log(denominator))
    
    if len(infonces) == 0:
        return torch.tensor(0.0, requires_grad=True).to(embeds.device)
    
    contrastive_loss = torch.mean(torch.stack(infonces)) * contrastive_weight
    return contrastive_loss


def calc_ranking_loss(embeds, causal_effects, data_splits,
                      ranking_weight=1000.0, margin=0.001,
                      min_ce_diff=0.2, max_ce_diff=0.5):
    """
    Ranking loss: embeddings should reflect causal effect ordering.
    
    If intervention A has higher causal effect than intervention B,
    then embed_A should be farther from the factual than embed_B.
    
    Args:
        embeds: (N_total,) tensor of L2-normalized embeddings
        causal_effects: list of tensors per sample group
        data_splits: list of start indices for each group
        ranking_weight: scaling factor
        margin: margin for MarginRankingLoss
        min_ce_diff: minimum causal effect difference to form a valid pair
        max_ce_diff: maximum causal effect difference to form a valid pair
    
    Returns:
        ranking_loss: scalar tensor
    """
    ranking_losses = []
    
    for sample_id in range(len(causal_effects)):
        ce = causal_effects[sample_id]
        if len(ce) < 2:
            continue
            
        # Factual embedding
        q = embeds[data_splits[sample_id]]
        
        # Counterfactual embeddings, sorted by causal effect
        sort_indices = torch.argsort(ce)
        keys = embeds[data_splits[sample_id] + 1:data_splits[sample_id + 1]]
        keys = keys[sort_indices]
        
        # Similarity to factual (higher = closer)
        dists = torch.matmul(keys, q)
        
        # Sorted causal effects
        ce_sorted = ce[sort_indices]
        
        # Build difference matrix to find valid pairs
        ce_row = ce_sorted.unsqueeze(1)
        ce_col = ce_sorted.unsqueeze(0)
        diff_ce = ce_col - ce_row  # diff_ce[i,j] = ce[j] - ce[i]
        
        # Valid pairs: j has higher CE than i by [min_ce_diff, max_ce_diff]
        mask = (diff_ce >= min_ce_diff) & (diff_ce <= max_ce_diff)
        
        if mask.sum() == 0:
            continue
        
        # Sample one valid pair per row that has valid columns
        lengths = mask.sum(dim=1)
        valid_rows = torch.where(lengths > 0)[0]
        
        if len(valid_rows) == 0:
            continue
        
        # For each valid row, pick a random valid column
        selected_pairs = []
        for row_idx in valid_rows:
            valid_cols = torch.where(mask[row_idx])[0]
            col_idx = valid_cols[torch.randint(len(valid_cols), (1,)).item()]
            selected_pairs.append((row_idx, col_idx))
        
        if len(selected_pairs) == 0:
            continue
        
        # MarginRankingLoss: dist[low_ce] should be > dist[high_ce]
        # i.e., low causal effect → closer to factual
        low_ce_dists = torch.stack([dists[p[0]] for p in selected_pairs])
        high_ce_dists = torch.stack([dists[p[1]] for p in selected_pairs])
        target = torch.ones(len(selected_pairs)).to(embeds.device)
        
        loss = F.margin_ranking_loss(low_ce_dists, high_ce_dists, target, margin=margin)
        ranking_losses.append(loss)
    
    if len(ranking_losses) == 0:
        return torch.tensor(0.0, requires_grad=True).to(embeds.device)
    
    ranking_loss = torch.mean(torch.stack(ranking_losses)) * ranking_weight
    return ranking_loss


def calc_causal_awareness_metrics(embeds, causal_effects, data_splits,
                                  non_causal_thresh=0.02, causal_thresh=0.1):
    """
    Evaluate causal awareness: HNC (Hallucinated Non-Causal) and ARS (Average Response Sensitivity).
    
    HNC: counts how many non-causal interventions the model incorrectly responds to
    ARS: average embedding distance for non-causal interventions (should be ~0)
    
    Args:
        embeds: L2-normalized embeddings
        causal_effects: ground truth causal effects
        data_splits: group boundaries
    
    Returns:
        hnc: int, number of hallucinated non-causal responses
        ars_values: list of sensitivity scores for non-causal interventions
    """
    hnc = 0
    ars_values = []
    
    with torch.no_grad():
        for sample_id in range(len(causal_effects)):
            q = embeds[data_splits[sample_id]]
            keys = embeds[data_splits[sample_id] + 1:data_splits[sample_id + 1]]
            
            ce = causal_effects[sample_id].cpu().numpy()
            
            # Embedding distances (1 - cosine similarity)
            sensitivities = 1.0 - torch.matmul(keys, q).cpu().numpy()
            
            # Non-causal interventions that model is sensitive to → hallucination
            non_causal_mask = ce < non_causal_thresh
            hnc += (sensitivities[non_causal_mask] > causal_thresh).sum()
            ars_values.extend(sensitivities[non_causal_mask].tolist())
    
    return hnc, ars_values
