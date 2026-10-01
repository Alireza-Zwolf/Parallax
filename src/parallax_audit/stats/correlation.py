"""Cross-method correlation between embedding and judge deviation (rebuttal).

Per-item Spearman rank correlation between the TARGET model's per-question
embedding deviation (paper Definition 2.1) and its judge score, computed
over the target's own N items only -- never pooled across all models in the
pool. The published rebuttal table (CS1=+0.52, CS2=+0.45, CS3=+0.85) uses
the target's RAW ordinal judge score ``s_i^T``, not the leave-one-out judge
deviation ``d_i``; the two coincide only when the target's peer median is
constant (as happens to be the case for CS3 at scale 10), so both are
exposed here via ``judge_value``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CorrelationResult:
    """Spearman correlation between a target's embedding and judge deviation."""

    rho: float
    p_value: float
    n: int
    judge_value: Literal["raw", "deviation"]
    case_study: str | None = None
    judge: str | None = None
    encoder: str | None = None
    scale: int | None = None


def spearman_cross_method(
    embedding_deviation: np.ndarray,
    target_scores: np.ndarray,
    peer_scores: Mapping[str, np.ndarray] | None = None,
    *,
    judge_value: Literal["raw", "deviation"] = "raw",
    case_study: str | None = None,
    judge: str | None = None,
    encoder: str | None = None,
    scale: int | None = None,
) -> CorrelationResult:
    """Spearman rho between the target's embedding deviation and judge score.

    ``embedding_deviation`` is the target's per-question embedding deviation
    (e.g. from ``parallax_audit.stats.deviation.deviation_vector``), length N.
    ``target_scores`` is the target's raw per-question judge score, also
    length N. With ``judge_value='raw'`` (the default, and what reproduces
    the published rebuttal numbers) the correlation uses ``target_scores``
    directly; with ``judge_value='deviation'`` it instead uses the
    leave-one-out judge deviation ``s_i^T - median_{k in peer_scores}(s_i^k)``,
    which requires ``peer_scores`` (a mapping of baseline key -> length-N
    score array, target excluded).

    ``case_study``, ``judge``, ``encoder`` and ``scale`` are carried through
    purely as labels for downstream reporting -- this function never assumes
    a fixed encoder or scale.
    """
    embedding_deviation = np.asarray(embedding_deviation, dtype=float)
    target_scores = np.asarray(target_scores, dtype=float)

    if judge_value == "raw":
        y = target_scores
    elif judge_value == "deviation":
        if not peer_scores:
            raise ValueError("judge_value='deviation' requires non-empty peer_scores")
        peer_matrix = np.stack([np.asarray(v, dtype=float) for v in peer_scores.values()])
        y = target_scores - np.median(peer_matrix, axis=0)
    else:
        raise ValueError(f"unknown judge_value {judge_value!r}")

    if embedding_deviation.shape != y.shape:
        raise ValueError("embedding_deviation and the judge values must be aligned")

    rho, p_value = stats.spearmanr(embedding_deviation, y)
    return CorrelationResult(
        rho=float(rho),
        p_value=float(p_value),
        n=len(y),
        judge_value=judge_value,
        case_study=case_study,
        judge=judge,
        encoder=encoder,
        scale=scale,
    )
