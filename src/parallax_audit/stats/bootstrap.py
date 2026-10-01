"""Nonparametric bootstrap test for LLM-as-a-Judge deviation (Section 2.3.3).

``H0: D_T - D_B <= 0`` vs. ``H1: D_T - D_B > 0``. Questions (or, with
``unit='topic'``, whole topics as intact blocks) are the resampling unit:
each replicate draws one set of indices and recomputes BOTH ``D_T`` and
``D_B`` from it, so noise shared between the two point estimates cancels
inside each replicate rather than being double-counted.

Unlike the embedding Welch test, ``D_B`` here really is defined by the paper
as the pooled-then-median statistic (paper Definition 2.2): pool every
baseline's per-question deviation-against-its-fellow-baselines vector, then
take one median over the pooled array. That pooling is unaffected by the
Welch-side correction in ``parallax_audit.stats.welch`` -- see that module's
docstring for why the two statistics pool differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from .deviation import judge_deviation


@dataclass(frozen=True)
class BootstrapResult:
    """Result of the one-sided bootstrap test on judge-deviation medians."""

    target: str
    D_T: float
    D_B: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    B: int
    seed: int
    unit: Literal["question", "topic"]


def judge_bootstrap_test(
    scores: Mapping[str, np.ndarray],
    target: str,
    baselines: Sequence[str],
    *,
    unit: Literal["question", "topic"] = "question",
    topics: Sequence | None = None,
    B: int = 20000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapResult:
    """Bootstrap ``D_T - D_B`` and report the point estimate, CI and p-value.

    The p-value uses the standard ``(#{delta* <= 0} + 1) / (B + 1)``
    estimator, which is never exactly zero (this reproduces the paper's
    ``p=5.0e-5`` floor at ``B=20000``). Deterministic for a fixed ``seed``,
    via ``np.random.default_rng``.
    """
    if B < 1 or not 0 < ci < 1:
        raise ValueError("B must be positive and ci must lie between 0 and 1")
    if unit == "topic" and (topics is None or len(topics) != len(scores[target])):
        raise ValueError("topics must match question count")
    baselines = list(baselines)
    if len(baselines) < 2:
        raise ValueError("need at least 2 baselines to leave one out")

    point = judge_deviation(scores, target, baselines)
    delta_obs = point.D_T - point.D_B

    rng = np.random.default_rng(seed)
    n = len(np.asarray(scores[target]))

    if unit == "question":
        deltas = _bootstrap_deltas_by_question(scores, target, baselines, n, B, rng)
    elif unit == "topic":
        deltas = _bootstrap_deltas_by_topic(scores, target, baselines, topics, B, rng)
    else:
        raise ValueError(f"unknown unit {unit!r}")

    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    ci_low, ci_high = np.percentile(deltas, [lo, hi])
    p_value = (int(np.sum(deltas <= 0)) + 1) / (B + 1)

    return BootstrapResult(
        target=target,
        D_T=point.D_T,
        D_B=point.D_B,
        delta=float(delta_obs),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        B=B,
        seed=seed,
        unit=unit,
    )


def _bootstrap_deltas_by_question(
    scores: Mapping[str, np.ndarray],
    target: str,
    baselines: Sequence[str],
    n: int,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised question-level bootstrap: all ``B`` replicates at once.

    Every replicate resamples the same N question indices (with
    replacement) for the target and every baseline, so ``D_T`` and ``D_B``
    are recomputed on matched data each time.
    """
    idx = rng.integers(0, n, size=(B, n))  # (B, n), shared across models

    target_r = np.asarray(scores[target], dtype=float)[idx]  # (B, n)
    baseline_r = np.stack(
        [np.asarray(scores[b], dtype=float)[idx] for b in baselines]
    )  # (P, B, n)

    d_target = target_r - np.median(baseline_r, axis=0)  # (B, n)
    D_T = np.median(d_target, axis=1)  # (B,)

    p = len(baselines)
    d_baseline_parts = []
    for i in range(p):
        others = [j for j in range(p) if j != i]
        peer_median = np.median(baseline_r[others], axis=0)  # (B, n)
        d_baseline_parts.append(baseline_r[i] - peer_median)  # (B, n)
    d_baseline = np.concatenate(d_baseline_parts, axis=1)  # (B, p*n) -- pooled, per Def 2.2
    D_B = np.median(d_baseline, axis=1)  # (B,)

    return D_T - D_B


def _bootstrap_deltas_by_topic(
    scores: Mapping[str, np.ndarray],
    target: str,
    baselines: Sequence[str],
    topics: Sequence,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cluster bootstrap: whole topics resampled as intact blocks.

    Topics vary in question count, so (unlike the question-level path) the
    resampled index set's composition differs per replicate; this stays a
    plain loop over replicates for clarity.
    """
    topics = np.asarray(topics)
    unique_topics = np.unique(topics)
    index_by_topic = {t: np.flatnonzero(topics == t) for t in unique_topics}

    deltas = np.empty(B)
    for r in range(B):
        chosen = rng.choice(unique_topics, size=len(unique_topics), replace=True)
        idx = np.concatenate([index_by_topic[t] for t in chosen])
        resampled = {m: np.asarray(v, dtype=float)[idx] for m, v in scores.items()}
        jd = judge_deviation(resampled, target, baselines)
        deltas[r] = jd.D_T - jd.D_B
    return deltas
