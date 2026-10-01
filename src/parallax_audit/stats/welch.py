"""One-sided Welch t-test for embedding deviation (paper Section 2.3.3).

``H0: mu_T - mu_B <= 0`` vs. ``H1: mu_T - mu_B > 0``, alpha = 0.05.

Two conventions matter here and are easy to conflate -- both are pinned by
regression tests in ``tests/test_stats.py``:

1. **Leave-target-out.** ``mu_T`` is the target's per-question deviation
   against ALL baselines. ``mu_B`` is built from each baseline's deviation
   against the OTHER BASELINES ONLY -- the target must never appear in any
   baseline's peer set, or ``mu_B`` is inflated.
2. **Per-question mean, not pooled.** ``mu_B`` is the per-question MEAN
   ACROSS BASELINES of those (K-1) length-N vectors, itself a length-N
   vector -- *not* the (K-1)*N concatenation of them. Concatenating treats
   strongly correlated baseline observations (they share the same questions)
   as independent, which shrinks the Welch standard error and inflates
   |t| relative to the paper's published statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import stats

from .deviation import EmbeddingSet, aggregate_by_topic, deviation_vector


@dataclass(frozen=True)
class WelchResult:
    """Result of the one-sided Welch t-test on embedding deviation."""

    target: str
    mu_t: float
    mu_b: float
    diff: float
    t_stat: float
    p_value: float
    n: int  # length of the (aligned) mu_t / mu_b vectors fed to the t-test
    unit: Literal["question", "topic"]


def embedding_welch_test(
    dist: EmbeddingSet,
    target: str,
    baselines: Sequence[str],
    *,
    unit: Literal["question", "topic"] = "question",
    topics: Sequence | None = None,
) -> WelchResult:
    """One-sided Welch t-test on embedding deviation, leave-target-out.

    ``mu_T`` = ``deviation_vector(dist, target, baselines)`` (length N).
    ``mu_B`` = the per-question mean, across baselines, of each baseline's
    deviation against its fellow baselines (target excluded); also length N.
    With ``unit='topic'`` both length-N vectors are first averaged within
    each topic (rebuttal) before the same test is run on the
    resulting length-T vectors.
    """
    baselines = list(baselines)
    if len(baselines) < 2:
        raise ValueError("need at least 2 baselines to leave one out")

    mu_t_vec = deviation_vector(dist, target, baselines)
    baseline_matrix = np.stack(
        [deviation_vector(dist, b, [x for x in baselines if x != b]) for b in baselines]
    )  # (K-1, N)
    mu_b_vec = baseline_matrix.mean(axis=0)  # (N,) -- mean ACROSS baselines, not pooled

    if unit == "topic":
        if topics is None:
            raise ValueError("unit='topic' requires `topics`")
        mu_t_vec = aggregate_by_topic(mu_t_vec, topics)
        mu_b_vec = aggregate_by_topic(mu_b_vec, topics)
    elif unit != "question":
        raise ValueError(f"unknown unit {unit!r}")

    t_stat, p_value = stats.ttest_ind(mu_t_vec, mu_b_vec, equal_var=False, alternative="greater")
    return WelchResult(
        target=target,
        mu_t=float(mu_t_vec.mean()),
        mu_b=float(mu_b_vec.mean()),
        diff=float(mu_t_vec.mean() - mu_b_vec.mean()),
        t_stat=float(t_stat),
        p_value=float(p_value),
        n=len(mu_t_vec),
        unit=unit,
    )
