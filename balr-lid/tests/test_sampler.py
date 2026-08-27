from collections import Counter

import pytest

from balr_lid.data.sampler import BalancedBatchSampler


def make_labels():
    # 3 classes, deliberately imbalanced: 50 / 20 / 5
    return [0] * 50 + [1] * 20 + [2] * 5


def test_batch_mode_every_batch_is_balanced():
    labels = make_labels()
    sampler = BalancedBatchSampler(labels=labels, batch_size=6, balance_mode="batch", seed=0)

    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        counts = Counter(labels[i] for i in batch)
        assert set(counts.values()) == {2}  # batch_size=6 / 3 classes = 2 per class, every batch


def test_batch_mode_requires_divisible_batch_size():
    with pytest.raises(ValueError):
        BalancedBatchSampler(labels=make_labels(), batch_size=7, balance_mode="batch")


def test_epoch_mode_pool_is_class_balanced_overall():
    labels = make_labels()
    sampler = BalancedBatchSampler(labels=labels, batch_size=6, balance_mode="epoch", seed=0)

    batches = list(sampler)
    all_indices = [i for batch in batches for i in batch]
    counts = Counter(labels[i] for i in all_indices)

    # Individual batches may be uneven after the global shuffle, but the
    # class totals over the whole epoch must be (near-)equal.
    values = list(counts.values())
    assert max(values) - min(values) <= 1


def test_unbalanced_mode_covers_a_shuffled_slice_of_the_dataset():
    labels = make_labels()
    sampler = BalancedBatchSampler(labels=labels, batch_size=5, balance_mode="unbalanced", drop_last=True, seed=0)

    batches = list(sampler)
    total = sum(len(b) for b in batches)
    assert total <= len(labels)
    assert total % 5 == 0


def test_set_epoch_changes_batch_composition():
    labels = make_labels()
    sampler = BalancedBatchSampler(labels=labels, batch_size=6, balance_mode="batch", seed=0)

    first_epoch = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)

    assert first_epoch != second_epoch


def test_len_matches_iter_batch_count():
    labels = make_labels()
    sampler = BalancedBatchSampler(labels=labels, batch_size=6, balance_mode="epoch", seed=0)
    assert len(sampler) == len(list(sampler))
