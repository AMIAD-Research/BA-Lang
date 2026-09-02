"""Class-balanced batch sampler.

Two balance modes are supported :

- ``"batch"``: every single batch contains exactly the same number of
  examples per class. 

- ``"epoch"``: a balanced pool of examples (equal total count per class) is
  drawn once per epoch, globally shuffled, then cut into batches of
  `batch_size`. Individual batches may end up with an uneven class mix, but
  the epoch as a whole is not biased towards majority classes.

- ``"unbalanced"``: plain shuffled iteration, no resampling. Useful as a
  baseline/debugging mode.


"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional, Sequence

import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import Sampler

VALID_BALANCE_MODES = ("batch", "epoch", "unbalanced")


class BalancedBatchSampler(Sampler):
    """Yields lists of dataset indices (one list = one batch).

    Args:
        labels: class index for every dataset item, in dataset order
            (e.g. `LanguageIdDataset.labels`).
        batch_size: number of examples per batch. In `balance_mode="batch"`
            it must be divisible by the number of classes.
        balance_mode: one of "batch", "epoch", "unbalanced".
        drop_last: drop a final undersized batch (epoch/unbalanced modes).
        num_replicas / rank: for multi-GPU (DDP) training, so each process
            gets a disjoint shard of batches. Auto-detected from
            `torch.distributed` if not given.
        seed: base random seed; combined with the current epoch so that
            shuffling is different every epoch.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        balance_mode: str = "batch",
        drop_last: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        if balance_mode not in VALID_BALANCE_MODES:
            raise ValueError(f"balance_mode must be one of {VALID_BALANCE_MODES}, got {balance_mode!r}")

        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.balance_mode = balance_mode
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.num_replicas = num_replicas or torch.distributed.get_world_size()
            self.rank = rank if rank is not None else torch.distributed.get_rank()
        else:
            self.num_replicas = num_replicas or 1
            self.rank = rank or 0

        self.classes = sorted(set(self.labels))
        self.indices_by_class = {
            c: [i for i, label in enumerate(self.labels) if label == c] for c in self.classes
        }

        if self.balance_mode == "batch" and self.batch_size % len(self.classes) != 0:
            raise ValueError(
                f"balance_mode='batch' requires batch_size ({self.batch_size}) to be "
                f"divisible by the number of classes ({len(self.classes)})."
            )

        self._rng = random.Random(self.seed)
        self._deques: dict[int, deque] = {}
        self._refill_all_deques()

    def set_epoch(self, epoch: int) -> None:
        """Advance to a new epoch: reseeds shuffling deterministically per-epoch."""
        self.epoch = epoch
        self._rng = random.Random(self.seed + epoch)

    # -- internal: persistent per-class draw queues -------------------------

    def _refill_deque(self, cls: int) -> None:
        pool = list(self.indices_by_class[cls])
        self._rng.shuffle(pool)
        self._deques[cls] = deque(pool)

    def _refill_all_deques(self) -> None:
        for c in self.classes:
            self._refill_deque(c)

    def _draw(self, cls: int, n: int) -> list[int]:
        out = []
        for _ in range(n):
            if not self._deques[cls]:
                self._refill_deque(cls)
            out.append(self._deques[cls].popleft())
        return out

    # -- batch count, computed without mutating sampler state ---------------

    def _num_batches(self) -> int:
        if self.balance_mode == "batch":
            samples_per_class = self.batch_size // len(self.classes)
            min_class_size = min(len(v) for v in self.indices_by_class.values())
            n = min_class_size // samples_per_class
        elif self.balance_mode == "epoch":
            min_class_size = min(len(v) for v in self.indices_by_class.values())
            pool_size = min_class_size * len(self.classes)
            n = pool_size // self.batch_size
        else:
            n = len(self.labels) // self.batch_size

        n = n // self.num_replicas
        return max(1, n)

    # -- batch generation (mutates the per-class deques) ---------------------

    def _generate_batch_mode(self, num_batches: int) -> list[list[int]]:
        samples_per_class = self.batch_size // len(self.classes)
        batches = []
        for _ in range(num_batches):
            batch = []
            for c in self.classes:
                batch.extend(self._draw(c, samples_per_class))
            self._rng.shuffle(batch)
            batches.append(batch)
        return batches

    def _generate_epoch_mode(self, num_batches: int) -> list[list[int]]:
        total = num_batches * self.batch_size
        needed_per_class = -(-total // len(self.classes))  # ceil division

        pool = []
        for c in self.classes:
            pool.extend(self._draw(c, needed_per_class))
        self._rng.shuffle(pool)

        pool = pool[: num_batches * self.batch_size]
        return [pool[i : i + self.batch_size] for i in range(0, len(pool), self.batch_size)]

    def _generate_unbalanced(self, num_batches: int) -> list[list[int]]:
        indices = list(range(len(self.labels)))
        self._rng.shuffle(indices)
        indices = indices[: num_batches * self.batch_size]
        return [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

    def __iter__(self):
        num_batches = self._num_batches()
        if self.balance_mode == "batch":
            batches = self._generate_batch_mode(num_batches)
        elif self.balance_mode == "epoch":
            batches = self._generate_epoch_mode(num_batches)
        else:
            batches = self._generate_unbalanced(num_batches)

        # Distributed sharding: each rank gets a disjoint slice of batches.
        batches = batches[self.rank :: self.num_replicas] if self.num_replicas > 1 else batches
        return iter(batches)

    def __len__(self) -> int:
        n = self._num_batches()
        return n // self.num_replicas if self.num_replicas > 1 else n


class SamplerEpochCallback(Callback):
    """Rotates a BalancedBatchSampler's internal state at the start of every epoch.

    `BalancedBatchSampler` holds persistent per-class draw queues so that,
    across epochs, majority-class examples are cycled through rather than
    reshuffled from scratch (which would bias re-sampling towards the first
    few items). This callback is what tells it a new epoch has started; it
    is required whenever `BalancedBatchSampler` is used as a `batch_sampler`.
    """

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        sampler = getattr(trainer.train_dataloader, "batch_sampler", None)
        if isinstance(sampler, BalancedBatchSampler):
            sampler.set_epoch(trainer.current_epoch)
