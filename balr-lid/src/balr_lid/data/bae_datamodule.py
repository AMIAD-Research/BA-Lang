"""LightningDataModule wiring `EmbeddingH5Dataset` for BAE training.

"""
from __future__ import annotations

from typing import Optional

import hydra
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class BAEDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: DictConfig,
        val_dataset: DictConfig,
        test_dataset: Optional[DictConfig] = None,
        batch_size: int = 256,
        val_batch_size: int = 256,
        num_workers: int = 8,
    ):
        super().__init__()
        self._train_dataset_cfg = train_dataset
        self._val_dataset_cfg = val_dataset
        self._test_dataset_cfg = test_dataset
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = hydra.utils.instantiate(self._train_dataset_cfg)
        if self.val_dataset is None:
            self.val_dataset = hydra.utils.instantiate(self._val_dataset_cfg)
        if self.test_dataset is None and self._test_dataset_cfg is not None:
            self.test_dataset = hydra.utils.instantiate(self._test_dataset_cfg)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._num_workers,
            pin_memory=True,
            persistent_workers=self._num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self._val_batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
            persistent_workers=self._num_workers > 0,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self._val_batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
        )
