import pytest
import torch

from balr_lid.models.head import LanguageClassificationHead


def make_inputs(batch=2, time=20, dim=16, layers=3):
    prepooling = torch.randn(batch, time, dim, layers)
    padding_mask = torch.ones(batch, time, dtype=torch.bool)
    padding_mask[0, 15:] = False  # first example has trailing padding
    return prepooling, padding_mask


def test_cosine_head_output_shapes():
    head = LanguageClassificationHead(num_layers=3, input_dim=16, embed_dim=8, num_classes=5)
    prepooling, mask = make_inputs()

    out = head.forward_classification(prepooling, padding_mask=mask)

    assert out["logits"].shape == (2, 5)
    assert out["embeddings"].shape == (2, 8)


def test_balr_head_output_shapes_and_binarized_embedding():
    head = LanguageClassificationHead(
        num_layers=3,
        input_dim=16,
        embed_dim=8,
        num_classes=5,
        use_binary_encoder=True,
        binary_dim=6,
        use_balr_head=True,
    )
    prepooling, mask = make_inputs()

    out = head.forward_classification(prepooling, padding_mask=mask)

    assert out["logits"].shape == (2, 5)
    assert out["embeddings"].shape == (2, 6)
    assert set(torch.unique(out["embeddings"]).tolist()) <= {0.0, 1.0}


def test_balr_head_requires_binary_encoder():
    with pytest.raises(ValueError):
        LanguageClassificationHead(
            num_layers=3, input_dim=16, embed_dim=8, num_classes=5, use_balr_head=True
        )


def test_binary_encoder_requires_binary_dim():
    with pytest.raises(ValueError):
        LanguageClassificationHead(
            num_layers=3, input_dim=16, embed_dim=8, num_classes=5, use_binary_encoder=True
        )


def test_bit_dropout_requires_balr_head():
    with pytest.raises(ValueError):
        LanguageClassificationHead(
            num_layers=3, input_dim=16, embed_dim=8, num_classes=5,
            use_binary_encoder=True, binary_dim=6, bit_dropout=True,
        )


def make_balr_head(**dropout_kwargs):
    return LanguageClassificationHead(
        num_layers=3, input_dim=16, embed_dim=8, num_classes=5,
        use_binary_encoder=True, binary_dim=6, use_balr_head=True,
        bit_dropout=True, **dropout_kwargs,
    )


def test_bit_dropout_mask_respects_n_min_and_binary_dim():
    head = make_balr_head(dropout_n_min=2, dropout_seed=0)
    n_active = head._sample_dropout_n_active(batch_size=200)
    assert min(n_active) >= 2
    assert max(n_active) <= 6


def test_bit_dropout_geometric_law_favors_short_prefixes():
    head = make_balr_head(dropout_law="geometric", dropout_rho=0.1, dropout_n_min=1, dropout_seed=0)
    n_active = head._sample_dropout_n_active(batch_size=500)
    assert sum(n_active) / len(n_active) < 2.0  # low rho -> mostly near n_min=1


def test_bit_dropout_noop_in_eval_mode():
    head = make_balr_head(dropout_n_min=1, dropout_seed=0)
    head.eval()
    prepooling, mask = make_inputs()

    with torch.no_grad():
        out_a = head.forward_classification(prepooling, padding_mask=mask)
        out_b = head.forward_classification(prepooling, padding_mask=mask)

    assert torch.allclose(out_a["logits"], out_b["logits"])


def test_nested_dropout_changes_the_score():
    torch.manual_seed(0)
    x = torch.randint(0, 2, (4, 6)).float()

    head_matryoshka = make_balr_head(dropout_n_min=1, nested_dropout=False, dropout_seed=0)
    head_nested = make_balr_head(dropout_n_min=1, nested_dropout=True, dropout_seed=0)
    with torch.no_grad():
        head_nested.ba_logits.copy_(head_matryoshka.ba_logits)

    head_matryoshka.train()
    head_nested.train()
    with torch.no_grad():
        scores_matryoshka = head_matryoshka._balr_logits(x)
        scores_nested = head_nested._balr_logits(x)

    assert not torch.allclose(scores_matryoshka, scores_nested)
