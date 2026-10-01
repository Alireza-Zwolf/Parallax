"""Permute-the-target ablation (paper Tables 5, 6).

Re-runs the relevant hypothesis test with EACH model in a universe treated
as the target in turn, the remaining models as its baseline ensemble. Works
for both the embedding Welch test and the judge bootstrap test; both return
a tidy DataFrame, one row per candidate target, sorted by ascending p-value.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .bootstrap import judge_bootstrap_test
from .deviation import EmbeddingSet
from .welch import embedding_welch_test


def permute_embedding_targets(
    dist: EmbeddingSet,
    universe: Sequence[str],
    *,
    unit: Literal["question", "topic"] = "question",
    topics: Sequence | None = None,
) -> pd.DataFrame:
    """Embedding Welch test with each model in ``universe`` as target in turn.

    For each candidate, the rest of ``universe`` is its baseline ensemble.
    Returns one row per candidate, sorted by ascending p-value -- the model
    the test flags most strongly as a deviating target appears first.
    """
    rows = []
    for candidate in universe:
        baselines = [m for m in universe if m != candidate]
        result = embedding_welch_test(dist, candidate, baselines, unit=unit, topics=topics)
        rows.append(dataclasses.asdict(result))
    return pd.DataFrame(rows).sort_values("p_value", kind="stable").reset_index(drop=True)


def permute_judge_targets(
    scores: Mapping[str, np.ndarray],
    universe: Sequence[str],
    *,
    unit: Literal["question", "topic"] = "question",
    topics: Sequence | None = None,
    B: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """Judge-bootstrap analogue of :func:`permute_embedding_targets`."""
    rows = []
    for candidate in universe:
        baselines = [m for m in universe if m != candidate]
        result = judge_bootstrap_test(
            scores, candidate, baselines, unit=unit, topics=topics, B=B, seed=seed
        )
        rows.append(dataclasses.asdict(result))
    return pd.DataFrame(rows).sort_values("p_value", kind="stable").reset_index(drop=True)
