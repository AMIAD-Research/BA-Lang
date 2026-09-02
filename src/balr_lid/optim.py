"""Utilities for selectively freezing/unfreezing model parameters by name.

There are two stages in the training recipe :

    stage 1 (head only):      freeze_exclude_patterns: ["head"]
    stage 2 (backbone+head):  freeze_exclude_patterns: ["backbone", "head"]

A parameter stays trainable (`requires_grad=True`) if its fully-qualified
name matches at least one of `exclude_patterns` (regex search, not full
match); every other parameter is frozen.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import torch.nn as nn


def freeze_by_pattern(model: nn.Module, exclude_patterns: Optional[Sequence[str]]) -> list[str]:
    """Freeze all parameters of `model` except those matching `exclude_patterns`.

    Args:
        model: module to (un)freeze in place.
        exclude_patterns: list of regexes; a parameter is left trainable if
            its fully-qualified name matches any of them. `None` or an
            empty list freezes everything.

    Returns:
        Names of the parameters left trainable (for logging/debugging).
    """
    trainable = []
    for name, param in model.named_parameters():
        keep_trainable = bool(exclude_patterns) and any(re.search(pattern, name) for pattern in exclude_patterns)
        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable.append(name)
    return trainable
