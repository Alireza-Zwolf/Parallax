"""Deviation scores: embedding (Def 2.1) and LLM-as-a-Judge (Def 2.2).

Both scores share one idea -- how far a model sits from a peer group on a
per-question basis, aggregated afterwards. The embedding side works on
cosine distance between response embeddings; the judge side works on ordinal
scores and uses the median (robust to a single miscalibrated judge score) in
place of a mean.

Everything here takes plain ``numpy`` arrays / mappings of them, never a
``parallax_audit.data`` object, so it can be unit-tested with hand-built fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Embedding deviation (paper Definition 2.1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingSet:
    """Precomputed pairwise cosine-distance tensor for a fixed set of models.

    ``distances[j, k, i]`` is the cosine distance between model ``keys[j]``'s
    and model ``keys[k]``'s embedding of question ``i``. Building this once
    with :func:`build_distance_tensor` and then indexing into it per subset
    (rather than recomputing distances every time) is what makes the
    pool-growth sweep in ``parallax_audit.stats.pools`` tractable.
    """

    keys: tuple[str, ...]
    distances: np.ndarray  # (K, K, N)

    def index(self, key: str) -> int:
        """Position of ``key`` along the model axes."""
        try:
            return self.keys.index(key)
        except ValueError:
            raise KeyError(f"{key!r} not in embedding set {self.keys}") from None


def build_distance_tensor(embeddings: Mapping[str, np.ndarray]) -> EmbeddingSet:
    """Precompute the full pairwise cosine-distance tensor for ``embeddings``.

    ``embeddings`` maps model key -> ``(N, dim)`` array of per-question
    embeddings, aligned positionally across models (row ``i`` is the same
    question for every model). All models must cover the same N questions.
    """
    keys = tuple(embeddings)
    if not keys:
        raise ValueError("embeddings is empty")
    lengths = {np.asarray(embeddings[k]).shape[0] for k in keys}
    if len(lengths) != 1:
        raise ValueError(f"all models must share the question count, got {lengths}")

    x = np.stack([np.asarray(embeddings[k], dtype=float) for k in keys])  # (K, N, dim)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("embeddings must be finite two-dimensional arrays")
    norms = np.linalg.norm(x, axis=2, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero vectors have undefined cosine distance")
    x = x / norms
    similarity = np.einsum("jnd,knd->jkn", x, x)
    distances = 1.0 - np.clip(similarity, -1.0, 1.0)
    return EmbeddingSet(keys=keys, distances=distances)


def deviation_vector(dist: EmbeddingSet, target: str, peers: Sequence[str]) -> np.ndarray:
    """Per-question embedding deviation of ``target`` against ``peers``.

    ``delta(q_i, M_j) = (1 / |peers|) * sum_{k in peers} cos_dist(e_i^j, e_i^k)``
    (paper Definition 2.1). ``peers`` decides who counts as "the pool" -- the
    leave-target-out logic in ``parallax_audit.stats.welch`` relies on being able to
    pass a peer set that differs per model.
    """
    if len(peers) == 0:
        raise ValueError("need at least one peer")
    j = dist.index(target)
    peer_idx = [dist.index(p) for p in peers]
    return dist.distances[j, peer_idx, :].mean(axis=0)


def aggregate_by_topic(values: np.ndarray, topics: Sequence) -> np.ndarray:
    """Mean of ``values`` within each distinct topic (order of first appearance).

    Moves an array from the question unit to the topic unit before re-running
    a test (rebuttal): each output entry is one topic's average of
    the per-question values that fall in it.
    """
    values = np.asarray(values, dtype=float)
    topics = np.asarray(topics)
    if len(values) != len(topics):
        raise ValueError("values and topics must be the same length")
    unique_topics = pd.unique(topics)
    return np.array([values[topics == t].mean() for t in unique_topics])


# --------------------------------------------------------------------------
# LLM-as-a-Judge deviation (paper Definition 2.2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeDeviation:
    """Per-question and pooled judge deviations (paper Definition 2.2)."""

    d_target: np.ndarray  # (N,) target vs. median of all baselines
    d_baseline: np.ndarray  # (n_baselines * N,) pooled, target excluded throughout
    D_T: float
    D_B: float


def judge_deviation_vector(
    scores: Mapping[str, np.ndarray], target: str, peers: Sequence[str]
) -> np.ndarray:
    """``d_i = s_i^target - median_{k in peers}(s_i^k)`` (paper Definition 2.2).

    The median (not the mean) of the peer scores makes a single
    miscalibrated judge score unable to swing the deviation.
    """
    if len(peers) == 0:
        raise ValueError("need at least one peer")
    peer_scores = np.stack([np.asarray(scores[p], dtype=float) for p in peers])
    return np.asarray(scores[target], dtype=float) - np.median(peer_scores, axis=0)


def judge_deviation(
    scores: Mapping[str, np.ndarray], target: str, baselines: Sequence[str]
) -> JudgeDeviation:
    """Target and pooled-baseline judge deviations, target excluded throughout.

    ``d_target`` compares the target to the median of ALL baselines.
    ``d_baseline`` pools, over every baseline ``b``, its deviation from the
    median of the OTHER baselines -- the target is never a peer of any
    baseline. This mirrors the leave-target-out estimator used for the
    embedding Welch test (paper Section 2.3.3) and is what the paper's
    published ``D_B`` requires.
    """
    baselines = list(baselines)
    if len(baselines) < 2:
        raise ValueError("need at least 2 baselines to leave one out")
    d_target = judge_deviation_vector(scores, target, baselines)
    d_baseline = np.concatenate(
        [
            judge_deviation_vector(scores, b, [x for x in baselines if x != b])
            for b in baselines
        ]
    )
    return JudgeDeviation(
        d_target=d_target,
        d_baseline=d_baseline,
        D_T=float(np.median(d_target)),
        D_B=float(np.median(d_baseline)),
    )
