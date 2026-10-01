"""Inter-judge agreement (paper Table 2).

Quadratic-weighted Cohen's kappa over aligned ``(model, question)`` score
pairs from two judges. Quadratic weighting is degenerate for binary
(``scale=2``) labels: with only two categories every disagreement gets the
same (maximal) weight, so the weighted kappa collapses to the unweighted
value and can behave erratically near-chance. The paper marks these cells
with an asterisk and reports the exact agreement rate instead; this module
does the same and flags which statistic was used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import cohen_kappa_score


@dataclass(frozen=True)
class AgreementResult:
    """Inter-judge agreement for one (case study, scale) cell."""

    value: float
    method: Literal["quadratic_weighted_kappa", "exact_agreement"]
    n: int
    scale: int


def inter_judge_agreement(
    scores_a: np.ndarray, scores_b: np.ndarray, *, scale: int
) -> AgreementResult:
    """Agreement between two judges' scores on the same items (paper Table 2).

    ``scores_a`` and ``scores_b`` must be aligned by item (same model,
    question order). For ``scale == 2`` this returns the exact agreement
    rate; otherwise, quadratic-weighted Cohen's kappa.
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    if scores_a.shape != scores_b.shape:
        raise ValueError("scores_a and scores_b must be aligned, equal-length arrays")
    if scores_a.ndim != 1:
        raise ValueError("scores_a/scores_b must be 1-D")

    n = len(scores_a)
    if scale == 2:
        value = float(np.mean(scores_a == scores_b))
        method: Literal["quadratic_weighted_kappa", "exact_agreement"] = "exact_agreement"
    else:
        value = float(cohen_kappa_score(scores_a, scores_b, weights="quadratic"))
        method = "quadratic_weighted_kappa"
    return AgreementResult(value=value, method=method, n=n, scale=scale)
