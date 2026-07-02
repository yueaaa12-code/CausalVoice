"""
Causal Projector module for CausalVoice.

Inserts after the content encoder (enc_p) to produce low-dimensional,
L2-normalized embeddings used for causal contrastive/ranking regularization.

Mirrors the contrastive_projector in CausalSim2Real's AutoBotEgo model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalProjector(nn.Module):
    """
    Projects the content encoder's output into a compact causal embedding space.
    
    Architecture (following CausalSim2Real):
        enc_p output (hidden_channels × T) → temporal pooling → 
        Linear(input_dim, input_dim//8) → ReLU →
        Linear(input_dim//8, input_dim//64) → L2 normalize
    
    The output embedding captures content-level information that should be
    invariant to non-causal factors (noise, recording conditions, synthetic artifacts)
    but sensitive to causal factors (speaker identity, prosody).
    """
    
    def __init__(self, hidden_channels=192, temporal_pool='mean', proj_dim=None):
        """
        Args:
            hidden_channels: dimension of enc_p output (default 192 for OOVC)
            temporal_pool: 'mean', 'max', or 'attention' pooling over time
            proj_dim: output embedding dimension. If None, auto-computed.
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.temporal_pool = temporal_pool
        
        # Input dimension after temporal pooling
        input_dim = hidden_channels
        
        if proj_dim is None:
            proj_dim = max(input_dim // 64, 16)
        
        # Deep projector to learn non-linear causal structure from frozen encoder
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, proj_dim)
        )
        
        # Optional: attention-based temporal pooling
        if temporal_pool == 'attention':
            self.attn_weights = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 4),
                nn.Tanh(),
                nn.Linear(hidden_channels // 4, 1)
            )
    
    def pool_temporal(self, x, mask=None):
        """
        Pool over temporal dimension.
        
        Args:
            x: (B, H, T) encoder output
            mask: (B, 1, T) valid time steps mask
        
        Returns:
            pooled: (B, H)
        """
        if mask is not None:
            x = x * mask
        
        if self.temporal_pool == 'mean':
            if mask is not None:
                pooled = x.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            else:
                pooled = x.mean(dim=-1)
                
        elif self.temporal_pool == 'max':
            if mask is not None:
                x = x.masked_fill(mask == 0, float('-inf'))
            pooled = x.max(dim=-1)[0]
            
        elif self.temporal_pool == 'attention':
            # x: (B, H, T) → (B, T, H)
            x_t = x.transpose(1, 2)
            attn = self.attn_weights(x_t).squeeze(-1)  # (B, T)
            if mask is not None:
                attn = attn.masked_fill(mask.squeeze(1) == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1).unsqueeze(1)  # (B, 1, T)
            pooled = torch.bmm(attn, x_t).squeeze(1)  # (B, H)
        else:
            raise ValueError(f"Unknown temporal_pool: {self.temporal_pool}")
        
        return pooled
    
    def forward(self, encoder_output, mask=None):
        """
        Args:
            encoder_output: (B, hidden_channels, T) from enc_p
            mask: (B, 1, T) valid positions mask
        
        Returns:
            embed: (B, proj_dim) L2-normalized causal embedding
        """
        # Temporal pooling: (B, H, T) → (B, H)
        pooled = self.pool_temporal(encoder_output, mask)
        
        # Project to low-dimensional space
        embed = self.projector(pooled)
        
        # L2 normalize
        embed = F.normalize(embed, p=2, dim=1)
        
        return embed
