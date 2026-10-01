"""Baseline pools and the exhaustive pool-growth sweep (rebuttal).

``resolve_pool`` turns a raw pool member list (which may or may not list the
target -- see ``parallax_audit/config/pools.yaml``) into a clean baseline tuple.
``pool_growth_sweep`` exhaustively tests every ``C(len(universe), K)``
baseline subset for each pool size K, reusing one precomputed
``parallax_audit.stats.deviation.EmbeddingSet`` so the O(1000+) subsets cost only
an index-and-mean each rather than a fresh embedding-distance computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from .deviation import EmbeddingSet
from .welch import embedding_welch_test


def resolve_pool(pool_members: Sequence[str], target: str) -> tuple[str, ...]:
    """Baseline pool with ``target`` removed, regardless of whether it was listed.

    Raises ``ValueError`` if fewer than 2 baselines remain -- the
    leave-target-out estimator (see ``parallax_audit.stats.welch``) needs at least
    2 baselines to leave one out of.
    """
    if len(set(pool_members)) != len(pool_members):
        raise ValueError("pool members must be unique")
    baselines = tuple(m for m in pool_members if m != target)
    if len(baselines) < 2:
        raise ValueError(
            f"pool has only {len(baselines)} baseline(s) after removing "
            f"target {target!r}; need at least 2"
        )
    return baselines


@dataclass(frozen=True)
class PoolGrowthResult:
    """One row (one pool size K) of the exhaustive pool-growth sweep."""

    size: int
    n_subsets: int
    n_rejected: int
    fraction_rejected: float
    alpha: float


def pool_growth_sweep(
    dist: EmbeddingSet,
    target: str,
    universe: Sequence[str],
    sizes: Sequence[int],
    *,
    alpha: float = 0.05,
) -> list[PoolGrowthResult]:
    """Exhaustively test every size-K baseline subset of ``universe``, for each K.

    For each K in ``sizes``, enumerates all ``C(len(universe), K)`` subsets
    of ``universe`` (with ``target`` excluded first), runs
    ``embedding_welch_test`` for each, and reports the fraction of subsets
    with ``p < alpha``. ``dist`` should be built once via
    ``parallax_audit.stats.deviation.build_distance_tensor`` and reused across the
    whole sweep -- the expensive cosine-distance computation then happens a
    single time, not once per subset.
    """
    if not 0 < alpha < 1 or len(set(universe)) != len(universe):
        raise ValueError("alpha must lie in (0,1) and universe must be unique")
    candidates = [m for m in universe if m != target]
    results = []
    for k in sizes:
        if k < 2:
            raise ValueError("pool-growth sizes must be >= 2 (need 2 baselines to leave one out)")
        if k > len(candidates):
            raise ValueError("pool size exceeds available universe")
        n_subsets = 0
        n_rejected = 0
        for subset in combinations(candidates, k):
            test = embedding_welch_test(dist, target, subset)
            n_subsets += 1
            if test.p_value < alpha:
                n_rejected += 1
        results.append(
            PoolGrowthResult(
                size=k,
                n_subsets=n_subsets,
                n_rejected=n_rejected,
                fraction_rejected=n_rejected / n_subsets,
                alpha=alpha,
            )
        )
    return results
