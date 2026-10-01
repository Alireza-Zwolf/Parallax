"""Statistical core of the Parallax audit (paper Sections 2.3 and rebuttal).

Every function here takes plain ``numpy`` arrays, ``pandas`` DataFrames or
mappings of those -- never a path, a case-study key, or anything from
``parallax_audit.data`` -- so each statistic is independently unit-testable and
carries no I/O of its own. Results are frozen dataclasses with named fields,
never bare tuples or dicts.

Module map:

- ``deviation``   embedding deviation (Def 2.1) and judge deviation (Def 2.2)
- ``welch``       one-sided Welch t-test for embedding deviation (Sec 2.3.3)
- ``bootstrap``   nonparametric bootstrap test for judge deviation (Sec 2.3.3)
- ``agreement``   inter-judge agreement (Table 2)
- ``correlation`` cross-method Spearman correlation (rebuttal)
- ``permute``     permute-the-target ablation (Tables 5, 6)
- ``pools``       baseline pool resolution and the pool-growth sweep (rebuttal)

See ``tests/test_stats.py`` for fixtures with analytically known answers,
including regression tests pinning the two subtlest estimator choices: the
leave-target-out peer sets, and the per-question-mean (not pooled)
aggregation of the Welch test's ``mu_B``.
"""

from __future__ import annotations

from .agreement import AgreementResult, inter_judge_agreement
from .bootstrap import BootstrapResult, judge_bootstrap_test
from .correlation import CorrelationResult, spearman_cross_method
from .deviation import (
    EmbeddingSet,
    JudgeDeviation,
    aggregate_by_topic,
    build_distance_tensor,
    deviation_vector,
    judge_deviation,
    judge_deviation_vector,
)
from .permute import permute_embedding_targets, permute_judge_targets
from .pools import PoolGrowthResult, pool_growth_sweep, resolve_pool
from .welch import WelchResult, embedding_welch_test

__all__ = [
    "AgreementResult",
    "inter_judge_agreement",
    "BootstrapResult",
    "judge_bootstrap_test",
    "CorrelationResult",
    "spearman_cross_method",
    "EmbeddingSet",
    "JudgeDeviation",
    "aggregate_by_topic",
    "build_distance_tensor",
    "deviation_vector",
    "judge_deviation",
    "judge_deviation_vector",
    "permute_embedding_targets",
    "permute_judge_targets",
    "PoolGrowthResult",
    "pool_growth_sweep",
    "resolve_pool",
    "WelchResult",
    "embedding_welch_test",
]
