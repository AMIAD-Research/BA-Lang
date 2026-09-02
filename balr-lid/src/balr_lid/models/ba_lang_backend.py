"""BA-Lang backend: a closed-form Bernoulli Naive-Bayes classifier over
attribute vectors.

Scores each language by the same Bernoulli log-likelihood as the trainable
BA-LR head :

    score(x, c) = sum_k  x_k * log p[c, k] + (1 - x_k) * log(1 - p[c, k])
- `BALangBackend` computes `p[c, k]` as the *empirical*
    activation frequency of attribute `k` among enrollment utterances of
    language `c` -- a closed-form MLE. 

"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import torch


class BALangBackend:
    """
    Args:
        label_list: ordered list of language codes; `label_list[c]`
            corresponds to row `c` of `prob_tensor`.
        prob_tensor: (num_classes, n_attributes) tensor of activation
            probabilities, `prob_tensor[c, k]` = P(attribute k active |
            language c).
        epsilon: probabilities are clamped to `[epsilon, 1 - epsilon]`
            before taking logs.
    """

    def __init__(self, label_list: list[str], prob_tensor: torch.Tensor, epsilon: float = 1e-3):
        if prob_tensor.shape[0] != len(label_list):
            raise ValueError(
                f"prob_tensor has {prob_tensor.shape[0]} rows but label_list has {len(label_list)} entries"
            )
        self.label_list = list(label_list)
        self.prob_tensor = prob_tensor.float().clamp(epsilon, 1.0 - epsilon)
        self._log_p = torch.log(self.prob_tensor)
        self._log_1_minus_p = torch.log1p(-self.prob_tensor)

    @property
    def n_attributes(self) -> int:
        return self.prob_tensor.shape[1]

    @classmethod
    def from_activation_probabilities(
        cls, df: pd.DataFrame, attribute_prefix: str = "BA", epsilon: float = 1e-3
    ) -> "BALangBackend":
        """Builds a backend from a `language, {attribute_prefix}0, {attribute_prefix}1, ...` DataFrame.

        This is the format written by `balr_lid.train_ba_lang_backend`.
        """
        attribute_cols = sorted(
            (c for c in df.columns if c.startswith(attribute_prefix)),
            key=lambda c: int(c[len(attribute_prefix):]),
        )
        if not attribute_cols:
            raise ValueError(f"No columns starting with {attribute_prefix!r} found in {list(df.columns)}")

        label_list = df["language"].tolist()
        prob_tensor = torch.tensor(df[attribute_cols].to_numpy(), dtype=torch.float32)
        return cls(label_list, prob_tensor, epsilon=epsilon)

    @classmethod
    def from_csv(cls, path: str, attribute_prefix: str = "BA", epsilon: float = 1e-3) -> "BALangBackend":
        return cls.from_activation_probabilities(pd.read_csv(path), attribute_prefix=attribute_prefix, epsilon=epsilon)

    def _dim_slice(self, k_first: Optional[int], k_last: Optional[int]) -> slice:
        if k_first is not None and k_last is not None:
            raise ValueError("k_first and k_last are mutually exclusive -- pass only one")
        if k_first is not None:
            return slice(0, k_first)
        if k_last is not None:
            return slice(self.n_attributes - k_last, self.n_attributes)
        return slice(0, self.n_attributes)

    def log_likelihood(
        self, x: torch.Tensor, k_first: Optional[int] = None, k_last: Optional[int] = None
    ) -> torch.Tensor:
        """Bernoulli log-likelihood of binary attribute vectors `x` under each language.

        Args:
            x: (B, n_attributes) binary attribute vectors (values in {0, 1}).
            k_first: score using only the leading `k_first` attributes
                (dims `[0, k_first)`) instead of the full `n_attributes`.
            k_last: score using only the trailing `k_last` attributes
                (dims `[n_attributes - k_last, n_attributes)`). Mutually
                exclusive with `k_first`.

        Returns:
            (B, num_classes) log-likelihood matrix.
        """
        dims = self._dim_slice(k_first, k_last)
        x = x.float()[:, dims]
        return x @ self._log_p[:, dims].T + (1.0 - x) @ self._log_1_minus_p[:, dims].T

    def predict(
        self, x: torch.Tensor, k_first: Optional[int] = None, k_last: Optional[int] = None
    ) -> list[str]:
        """Argmax language prediction for each row of `x` (see `log_likelihood`)."""
        scores = self.log_likelihood(x, k_first=k_first, k_last=k_last)
        return [self.label_list[i] for i in scores.argmax(dim=-1).tolist()]
