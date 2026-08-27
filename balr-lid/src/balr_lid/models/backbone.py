"""Self-supervised speech backbone wrapper (wav2vec2 / MMS family).

Thin wrapper around a HuggingFace `Wav2Vec2Model` that adds:
  - padding-aware input normalization (mean/variance computed only over
    valid, non-padded samples, instead of the whole padded batch),
  - always returns *every* hidden-layer output (`output_hidden_states=True`),
    since the pooling head (see models/pooling.py) attends over all SSL
    layers, not just the last one,
  - the backbone's own reduced padding mask, mapping raw-audio padding to
    feature-frame padding.

Works with any HF checkpoint sharing the Wav2Vec2 architecture family,
including MMS (e.g. "facebook/mms-1b") and wav2vec2-XLS-R.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model as HFWav2Vec2Model


@dataclass
class BackboneOutput:
    layer_outputs: List[torch.Tensor]  # each (B, T', D); len == num_layers
    padding_mask: Optional[torch.Tensor]  # (B, T') bool, True = valid feature frame


class SSLBackbone(nn.Module):
    """Wraps a HuggingFace Wav2Vec2-family model.

    Args:
        pretrained_model_name_or_path: local directory or HF Hub id to load
            weights + config from (e.g. "facebook/mms-1b", or a local
            snapshot of it).
        normalize: zero-mean/unit-variance normalize the raw waveform before
            the backbone (standard wav2vec2 preprocessing), computed only
            over non-padded samples so it stays correct for batched
            variable-length audio.
    """

    def __init__(self, pretrained_model_name_or_path: str, normalize: bool = True):
        super().__init__()
        self.model = HFWav2Vec2Model.from_pretrained(pretrained_model_name_or_path)
        self.normalize = normalize

    @property
    def num_layers(self) -> int:
        """Number of hidden-state layers returned, including the input embeddings."""
        return self.model.config.num_hidden_layers + 1

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def normalize_with_padding(
        self, inputs: torch.Tensor, padding_mask: Optional[torch.Tensor], eps: float = 1e-7
    ) -> torch.Tensor:
        if padding_mask is None:
            mean = inputs.mean(dim=1, keepdim=True)
            variance = inputs.var(dim=1, keepdim=True)
            return (inputs - mean) / torch.sqrt(variance + eps)

        mask = padding_mask.to(inputs.dtype)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        masked = inputs * mask
        mean = masked.sum(dim=1, keepdim=True) / valid_counts
        variance = ((masked - mean) ** 2 * mask).sum(dim=1, keepdim=True) / valid_counts
        return (masked - mean) / torch.sqrt(variance + eps)

    def forward(self, inputs: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> BackboneOutput:
        """
        Args:
            inputs: (B, T) raw waveform at 16kHz.
            padding_mask: (B, T) bool, True = valid sample.
        """
        if self.normalize:
            inputs = self.normalize_with_padding(inputs, padding_mask)

        attention_mask = padding_mask.long() if padding_mask is not None else None
        outputs = self.model(
            inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        layer_outputs = list(outputs.hidden_states)

        feature_padding_mask = None
        if padding_mask is not None:
            feature_padding_mask = self.model._get_feature_vector_attention_mask(
                layer_outputs[0].shape[1], padding_mask, add_adapter=False
            ).bool()

        return BackboneOutput(layer_outputs=layer_outputs, padding_mask=feature_padding_mask)
