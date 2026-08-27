import torch.nn as nn

from balr_lid.optim import freeze_by_pattern


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


def test_freeze_head_only():
    model = _ToyModel()
    trainable = freeze_by_pattern(model, exclude_patterns=["head"])

    assert trainable and all(name.startswith("head") for name in trainable)
    assert not any(p.requires_grad for n, p in model.named_parameters() if n.startswith("backbone"))
    assert all(p.requires_grad for n, p in model.named_parameters() if n.startswith("head"))


def test_freeze_backbone_and_head_stage():
    model = _ToyModel()
    freeze_by_pattern(model, exclude_patterns=["backbone", "head"])

    assert all(p.requires_grad for p in model.parameters())


def test_no_patterns_freezes_everything():
    model = _ToyModel()
    trainable = freeze_by_pattern(model, exclude_patterns=None)

    assert trainable == []
    assert not any(p.requires_grad for p in model.parameters())
