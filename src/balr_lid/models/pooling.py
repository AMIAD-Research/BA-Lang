"""Multi-Head Factorized Attentive (MHFA) pooling over SSL layers and time.

Turns the stack of per-layer SSL hidden states into a single fixed-size
embedding by:
  1. collapsing the layer dimension with two *learned* softmax-weighted
     sums (one for "key", one for "value" — two separate learned mixtures
     of layers, following the standard weighted-layer-sum probing recipe),
  2. compressing the resulting per-frame vectors to `latent_dim`,
  3. computing multi-head attention weights over time from the compressed
     keys, masked so padded frames get zero weight,
  4. pooling values with those attention weights and averaging over heads.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadFactorizedAttentivePooling(nn.Module):
    """MHFA pooling: (B, D, T, L) SSL features -> (B, latent_dim) embedding.

    Args:
        num_layers: number of SSL hidden-state layers to pool over (must
            match the `L` dimension of the input, i.e. backbone.num_layers).
        input_dim: per-frame feature dimension `D` of the SSL backbone.
        latent_dim: output embedding dimension.
        num_heads: number of attention heads used for temporal pooling
            (their outputs are averaged, not concatenated).
    """

    def __init__(self, num_layers: int, input_dim: int, latent_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.layer_weights_key = nn.Parameter(torch.ones(num_layers))
        self.layer_weights_value = nn.Parameter(torch.ones(num_layers))

        self.compress_key = nn.Linear(input_dim, latent_dim)
        self.compress_value = nn.Linear(input_dim, latent_dim)
        self.attention_head = nn.Linear(latent_dim, num_heads)

    def forward(self, features: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (B, D, T, L) stacked SSL hidden states.
            padding_mask: (B, T) bool, True = valid frame.

        Returns:
            (B, latent_dim) pooled embedding.
        """
        key = torch.sum(features * F.softmax(self.layer_weights_key, dim=-1), dim=-1).transpose(1, 2)  # (B, T, D)
        value = torch.sum(features * F.softmax(self.layer_weights_value, dim=-1), dim=-1).transpose(1, 2)

        key = self.compress_key(key)  # (B, T, latent_dim)
        value = self.compress_value(value)

        attn_logits = self.attention_head(key)  # (B, T, num_heads)
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).expand_as(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=1)  # softmax over time

        pooled = torch.sum(value.unsqueeze(2) * attn_weights.unsqueeze(-1), dim=1)  # (B, num_heads, latent_dim)
        return pooled.mean(dim=1)  # (B, latent_dim)
