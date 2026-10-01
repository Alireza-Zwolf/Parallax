"""Unit tests for `parallax_audit.stats`.

Fixtures are hand-built so that the statistic under test can be computed on
paper (or with a two-line derivation given in a comment) and compared for
exact equality, rather than merely "runs without error". Two fixtures pin
the subtlest estimator choices in the module and must never regress:

- `test_leave_target_out_regression` -- including the target in a baseline's
  peer set inflates mu_B (paper Sec 2.3.3).
- `test_pooled_vs_per_question_mean_regression` -- pooling/concatenating the
  baselines' deviation vectors (instead of averaging them per question)
  inflates |t| relative to the paper's published Table 5/6 statistics.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from parallax_audit.stats.agreement import inter_judge_agreement
from parallax_audit.stats.bootstrap import judge_bootstrap_test
from parallax_audit.stats.correlation import spearman_cross_method
from parallax_audit.stats.deviation import (
    EmbeddingSet,
    aggregate_by_topic,
    build_distance_tensor,
    deviation_vector,
    judge_deviation,
)
from parallax_audit.stats.permute import permute_embedding_targets, permute_judge_targets
from parallax_audit.stats.pools import pool_growth_sweep, resolve_pool
from parallax_audit.stats.welch import embedding_welch_test

# ==========================================================================
# 1. Embedding deviation (Definition 2.1)
# ==========================================================================


def test_embedding_deviation_hand_computed_orthogonal_vectors():
    """N=1 question, 3 models, orthogonal embeddings -> exact cosine distances.

    e_A = [1,0], e_B = [0,1], e_C = [1,0]:
      cos_dist(A,B) = 1, cos_dist(A,C) = 0, cos_dist(B,C) = 1.
      delta(q1, A) = mean(1, 0) = 0.5
      delta(q1, B) = mean(1, 1) = 1.0
      delta(q1, C) = mean(0, 1) = 0.5
    """
    embeddings = {
        "A": np.array([[1.0, 0.0]]),
        "B": np.array([[0.0, 1.0]]),
        "C": np.array([[1.0, 0.0]]),
    }
    dist = build_distance_tensor(embeddings)

    assert deviation_vector(dist, "A", ["B", "C"]) == pytest.approx([0.5])
    assert deviation_vector(dist, "B", ["A", "C"]) == pytest.approx([1.0])
    assert deviation_vector(dist, "C", ["A", "B"]) == pytest.approx([0.5])

    assert deviation_vector(dist, "A", ["B", "C"]).mean() == pytest.approx(0.5)
    assert deviation_vector(dist, "B", ["A", "C"]).mean() == pytest.approx(1.0)


def test_embedding_deviation_identical_embeddings_is_zero():
    """All models identical -> every pairwise cosine distance is 0 -> D=0."""
    embeddings = {k: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) for k in "ABC"}
    dist = build_distance_tensor(embeddings)
    assert deviation_vector(dist, "A", ["B", "C"]).mean() == pytest.approx(0.0)


def test_build_distance_tensor_rejects_mismatched_question_counts():
    with pytest.raises(ValueError):
        build_distance_tensor({"A": np.zeros((3, 2)), "B": np.zeros((4, 2))})


def test_aggregate_by_topic_hand_computed():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    topics = ["a", "a", "b", "b"]
    assert aggregate_by_topic(values, topics) == pytest.approx([1.5, 3.5])


# ==========================================================================
# 2. Embedding hypothesis test (Welch, Section 2.3.3) + leave-target-out
# ==========================================================================


def _orthogonal_pair_fixture() -> EmbeddingSet:
    """K=4 (T, B1, B2, B3), N=1: T is orthogonal to identical baselines.

    dist(T, Bi) = 1 for all i; dist(Bi, Bj) = 0 for baseline pairs.
    Correct (leave-target-out) mu_B = 0 exactly, since every baseline's
    deviation against the OTHER baselines only involves zero distances.
    """
    keys = ("T", "B1", "B2", "B3")
    d = np.zeros((4, 4, 1))
    for i in range(1, 4):
        d[0, i, 0] = 1.0
        d[i, 0, 0] = 1.0
    return EmbeddingSet(keys=keys, distances=d)


def test_welch_leave_target_out_mu_b_hand_computed():
    dist = _orthogonal_pair_fixture()
    result = embedding_welch_test(dist, "T", ["B1", "B2", "B3"])
    assert result.mu_b == pytest.approx(0.0)
    assert result.mu_t == pytest.approx(1.0)
    assert result.n == 1


def test_leave_target_out_regression():
    """Pin the leave-target-out bug: including the target in a baseline's
    peer set must inflate mu_B relative to the correct (target-excluded)
    estimator. Uses the K=2-pool-plus-outlier fixture from Def 2.1 above.

    Correct mu_B = 0 (baselines are mutually identical). If a baseline's
    peer set incorrectly includes the target (dist=1 to an otherwise-0 peer
    set of size 2), its deviation becomes mean(0, 0, 1) = 1/3 > 0.
    """
    dist = _orthogonal_pair_fixture()
    baselines = ["B1", "B2", "B3"]

    correct = embedding_welch_test(dist, "T", baselines)

    buggy_vecs = [
        deviation_vector(dist, b, [x for x in baselines if x != b] + ["T"])
        for b in baselines
    ]
    buggy_mu_b = float(np.mean([v.mean() for v in buggy_vecs]))

    assert buggy_mu_b > correct.mu_b
    assert buggy_mu_b == pytest.approx(1.0 / 3.0)
    assert correct.mu_b == pytest.approx(0.0)


def test_pooled_vs_per_question_mean_regression():
    """Pin the pooling bug: concatenating the (K-1) baseline deviation
    vectors into one length-(K-1)*N sample (instead of averaging them per
    question into a length-N sample) treats correlated, same-question
    observations as independent and inflates |t|.

    Baselines share a common per-question shape (`common`) plus their own
    small idiosyncratic noise; the target is offset from that same shape by
    a constant shift. mu_T and mu_B (the means) are identical under both
    conventions -- only the variance/df, and hence t, differ.
    """
    n = 10
    common = np.linspace(0.05, 0.35, n)
    eps1 = 0.05 * np.sin(np.linspace(0, 3, n))
    eps2 = 0.05 * np.sin(np.linspace(1, 4, n))
    eps3 = 0.05 * np.sin(np.linspace(2, 5, n))

    dev_b1, dev_b2, dev_b3 = common + eps1, common + eps2, common + eps3
    # Solve the 3x3 linear system so each baseline's *leave-target-out*
    # deviation against its two fellow baselines equals the desired dev_bi.
    d12 = dev_b1 + dev_b2 - dev_b3
    d13 = dev_b1 + dev_b3 - dev_b2
    d23 = dev_b2 + dev_b3 - dev_b1
    mu_t_profile = common + 0.25  # target consistently above the common shape

    keys = ("T", "B1", "B2", "B3")
    idx = {k: i for i, k in enumerate(keys)}
    d = np.zeros((4, 4, n))
    d[idx["B1"], idx["B2"], :] = d12
    d[idx["B2"], idx["B1"], :] = d12
    d[idx["B1"], idx["B3"], :] = d13
    d[idx["B3"], idx["B1"], :] = d13
    d[idx["B2"], idx["B3"], :] = d23
    d[idx["B3"], idx["B2"], :] = d23
    for b in ("B1", "B2", "B3"):
        d[idx["T"], idx[b], :] = mu_t_profile
        d[idx[b], idx["T"], :] = mu_t_profile

    dist = EmbeddingSet(keys=keys, distances=d)
    baselines = ["B1", "B2", "B3"]

    correct = embedding_welch_test(dist, "T", baselines)

    mu_t_vec = deviation_vector(dist, "T", baselines)
    pooled_baseline = np.concatenate(
        [deviation_vector(dist, b, [x for x in baselines if x != b]) for b in baselines]
    )
    t_pooled, _ = scipy_stats.ttest_ind(
        mu_t_vec, pooled_baseline, equal_var=False, alternative="greater"
    )

    # Means agree under both conventions (only n/variance differ).
    assert correct.mu_b == pytest.approx(pooled_baseline.mean())
    # The correct, per-question-mean t must be smaller in magnitude.
    assert abs(correct.t_stat) < abs(t_pooled)


def test_welch_requires_at_least_two_baselines():
    dist = _orthogonal_pair_fixture()
    with pytest.raises(ValueError):
        embedding_welch_test(dist, "T", ["B1"])


def test_welch_k2_pool_runs():
    """Minimal K=2-baseline pool (3 models total) must not raise."""
    keys = ("T", "B1", "B2")
    d = np.zeros((3, 3, 5))
    rng = np.random.default_rng(0)
    for i in range(3):
        for j in range(i + 1, 3):
            vals = rng.uniform(0, 1, size=5)
            d[i, j, :] = vals
            d[j, i, :] = vals
    dist = EmbeddingSet(keys=keys, distances=d)
    result = embedding_welch_test(dist, "T", ["B1", "B2"])
    assert result.n == 5


def test_welch_topic_unit_shrinks_n_to_topic_count():
    dist = _pooling_fixture_10q()
    topics = ["x"] * 5 + ["y"] * 5
    result = embedding_welch_test(dist, "T", ["B1", "B2", "B3"], unit="topic", topics=topics)
    assert result.unit == "topic"
    assert result.n == 2


def _pooling_fixture_10q() -> EmbeddingSet:
    n = 10
    common = np.linspace(0.05, 0.35, n)
    baseline_noise = 0.01 * np.sin(np.linspace(0, 6, n))
    keys = ("T", "B1", "B2", "B3")
    idx = {k: i for i, k in enumerate(keys)}
    d = np.zeros((4, 4, n))
    d[idx["B1"], idx["B2"], :] = 0.1 + baseline_noise
    d[idx["B2"], idx["B1"], :] = 0.1 + baseline_noise
    d[idx["B1"], idx["B3"], :] = 0.1 - baseline_noise
    d[idx["B3"], idx["B1"], :] = 0.1 - baseline_noise
    d[idx["B2"], idx["B3"], :] = 0.1 + baseline_noise
    d[idx["B3"], idx["B2"], :] = 0.1 + baseline_noise
    for b in ("B1", "B2", "B3"):
        d[idx["T"], idx[b], :] = common + 0.3
        d[idx[b], idx["T"], :] = common + 0.3
    return EmbeddingSet(keys=keys, distances=d)


# ==========================================================================
# 3. LLM-as-a-Judge deviation (Definition 2.2)
# ==========================================================================


def test_judge_deviation_all_identical_scores_is_zero():
    scores = {k: np.array([3.0, 3.0, 3.0, 3.0]) for k in ("T", "B1", "B2", "B3")}
    jd = judge_deviation(scores, "T", ["B1", "B2", "B3"])
    assert jd.D_T == pytest.approx(0.0)
    assert jd.D_B == pytest.approx(0.0)


def test_judge_deviation_single_outlier_is_median_robust():
    """One baseline is a wild outlier; the median-based deviation must
    ignore it entirely, unlike a mean would."""
    scores = {"T": np.array([5.0]), "B1": np.array([5.0]), "B2": np.array([5.0]), "B3": np.array([100.0])}
    jd = judge_deviation(scores, "T", ["B1", "B2", "B3"])
    assert jd.D_T == pytest.approx(0.0)  # median([5,5,100]) = 5 = target score


def test_judge_deviation_hand_computed():
    """d_i = s_i^T - median_{k != T}(s_i^k); pooled D_B excludes target
    throughout, per-baseline peer sets."""
    scores = {
        "T": np.array([4.0, 6.0]),
        "B1": np.array([1.0, 5.0]),
        "B2": np.array([2.0, 5.0]),
        "B3": np.array([3.0, 7.0]),
    }
    jd = judge_deviation(scores, "T", ["B1", "B2", "B3"])
    # median(B1,B2,B3) per question: q1 -> median(1,2,3)=2, q2 -> median(5,5,7)=5
    assert jd.d_target == pytest.approx([4 - 2, 6 - 5])
    assert jd.D_T == pytest.approx(np.median([2.0, 1.0]))

    # B1 vs (B2,B3): median q1=median(2,3)=2.5, q2=median(5,7)=6
    # B2 vs (B1,B3): median q1=median(1,3)=2,   q2=median(5,7)=6
    # B3 vs (B1,B2): median q1=median(1,2)=1.5, q2=median(5,5)=5
    expected_pooled = np.array(
        [1 - 2.5, 5 - 6, 2 - 2, 5 - 6, 3 - 1.5, 7 - 5]
    )
    assert jd.d_baseline == pytest.approx(expected_pooled)
    assert jd.D_B == pytest.approx(np.median(expected_pooled))


def test_judge_deviation_requires_two_baselines():
    scores = {"T": np.array([1.0]), "B1": np.array([1.0])}
    with pytest.raises(ValueError):
        judge_deviation(scores, "T", ["B1"])


# ==========================================================================
# 4. Judge hypothesis test (nonparametric bootstrap on medians)
# ==========================================================================


def _separated_judge_fixture():
    n = 20
    target = 9.0 + np.tile([0.0, 1.0, 2.0, 1.0], n // 4)
    b1 = np.tile([0.0, 1.0, 2.0, 1.0], n // 4)
    b2 = np.tile([1.0, 0.0, 2.0, 0.0], n // 4)
    b3 = np.tile([2.0, 1.0, 0.0, 1.0], n // 4)
    return {"T": target, "B1": b1, "B2": b2, "B3": b3}


def test_bootstrap_deterministic_same_seed():
    scores = _separated_judge_fixture()
    r1 = judge_bootstrap_test(scores, "T", ["B1", "B2", "B3"], B=500, seed=0)
    r2 = judge_bootstrap_test(scores, "T", ["B1", "B2", "B3"], B=500, seed=0)
    assert r1 == r2


def test_bootstrap_different_seeds_can_differ():
    scores = _separated_judge_fixture()
    # Add mild variability so different resamples actually produce different
    # replicate arrays (the perfectly-separated fixture below is too rigid).
    rng = np.random.default_rng(7)
    scores = {k: v + rng.normal(0, 0.01, size=v.shape) for k, v in scores.items()}
    r_a = judge_bootstrap_test(scores, "T", ["B1", "B2", "B3"], B=500, seed=0)
    r_b = judge_bootstrap_test(scores, "T", ["B1", "B2", "B3"], B=500, seed=1)
    assert (r_a.ci_low, r_a.ci_high) != (r_b.ci_low, r_b.ci_high)


def test_bootstrap_p_value_floor_for_perfectly_separated_fixture():
    """Target always far above baselines for every question, regardless of
    resample composition -> no replicate can produce delta <= 0 -> the
    p-value hits the (0 + 1) / (B + 1) floor exactly."""
    scores = _separated_judge_fixture()
    B = 500
    result = judge_bootstrap_test(scores, "T", ["B1", "B2", "B3"], B=B, seed=0)
    assert result.p_value == pytest.approx(1.0 / (B + 1))
    assert result.delta > 0


def test_bootstrap_ci_brackets_point_estimate():
    scores = _separated_judge_fixture()
    result = judge_bootstrap_test(scores, "T", ["B1", "B2", "B3"], B=1000, seed=0)
    assert result.ci_low <= result.delta <= result.ci_high


def test_bootstrap_requires_two_baselines():
    scores = {"T": np.array([1.0, 2.0]), "B1": np.array([1.0, 2.0])}
    with pytest.raises(ValueError):
        judge_bootstrap_test(scores, "T", ["B1"], B=10)


def test_bootstrap_topic_unit_runs_and_brackets_point_estimate():
    scores = _separated_judge_fixture()
    topics = ["a"] * 10 + ["b"] * 10
    result = judge_bootstrap_test(
        scores, "T", ["B1", "B2", "B3"], unit="topic", topics=topics, B=200, seed=0
    )
    assert result.ci_low <= result.delta <= result.ci_high
    assert result.unit == "topic"


# ==========================================================================
# 5. Inter-judge agreement (Table 2)
# ==========================================================================


def test_kappa_perfect_agreement_is_one():
    a = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    result = inter_judge_agreement(a, a, scale=4)
    assert result.value == pytest.approx(1.0)
    assert result.method == "quadratic_weighted_kappa"


def test_kappa_binary_returns_exact_agreement_rate():
    a = np.array([0, 1, 0, 1, 1])
    b = np.array([0, 1, 1, 1, 0])
    result = inter_judge_agreement(a, b, scale=2)
    assert result.value == pytest.approx(0.6)  # 3/5 matches
    assert result.method == "exact_agreement"


def test_agreement_rejects_misaligned_lengths():
    with pytest.raises(ValueError):
        inter_judge_agreement(np.array([1, 2]), np.array([1, 2, 3]), scale=4)


# ==========================================================================
# 6. Cross-method correlation (rebuttal)
# ==========================================================================


def test_spearman_perfect_positive_and_negative():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pos = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y_neg = y_pos[::-1]
    assert spearman_cross_method(x, y_pos).rho == pytest.approx(1.0)
    assert spearman_cross_method(x, y_neg).rho == pytest.approx(-1.0)


def test_spearman_default_uses_raw_judge_score():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    result = spearman_cross_method(x, y)
    assert result.judge_value == "raw"
    assert result.n == 3


def test_spearman_raw_vs_deviation_variants_differ():
    """Pins the raw-score-vs-deviation distinction: on a fixture where the
    target's peer median is non-constant across questions, the two
    variants must give different rho."""
    embedding_deviation = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    target_scores = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    peer_scores = {
        "B1": np.array([1.0, 1.0, 1.0, 9.0, 9.0]),
        "B2": np.array([1.0, 9.0, 1.0, 1.0, 9.0]),
    }
    raw = spearman_cross_method(embedding_deviation, target_scores, peer_scores, judge_value="raw")
    dev = spearman_cross_method(embedding_deviation, target_scores, peer_scores, judge_value="deviation")
    assert raw.rho != pytest.approx(dev.rho)
    assert raw.judge_value == "raw"
    assert dev.judge_value == "deviation"


def test_spearman_deviation_requires_peer_scores():
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        spearman_cross_method(x, y, judge_value="deviation")


# ==========================================================================
# 7. Permute-the-target ablation (Tables 5, 6)
# ==========================================================================


def _strongly_separated_universe() -> EmbeddingSet:
    """Model X is far from everyone; A/B/C/D are mutually close."""
    keys = ("X", "A", "B", "C", "D")
    n = 6
    rng = np.random.default_rng(0)
    d = np.zeros((5, 5, n))
    for i in range(5):
        for j in range(i + 1, 5):
            if keys[i] == "X" or keys[j] == "X":
                vals = 0.8 + 0.02 * rng.standard_normal(n)
            else:
                vals = 0.05 + 0.01 * rng.standard_normal(n)
            d[i, j, :] = vals
            d[j, i, :] = vals
    return EmbeddingSet(keys=keys, distances=d)


def test_permute_embedding_targets_ranks_true_target_first():
    dist = _strongly_separated_universe()
    df = permute_embedding_targets(dist, list(dist.keys))
    assert df.iloc[0]["target"] == "X"
    assert (df["p_value"].values == np.sort(df["p_value"].values)).all()


def test_permute_judge_targets_ranks_true_target_first():
    n = 20
    rng = np.random.default_rng(0)
    baseline_common = np.tile([0.0, 1.0, 2.0, 1.0], n // 4)
    scores = {
        "T": baseline_common + 8.0,
        "A": baseline_common + rng.normal(0, 0.05, n),
        "B": baseline_common + rng.normal(0, 0.05, n),
        "C": baseline_common + rng.normal(0, 0.05, n),
    }
    df = permute_judge_targets(scores, list(scores), B=300, seed=0)
    assert df.iloc[0]["target"] == "T"


# ==========================================================================
# 8. Topic-level / cluster-robust tests -- covered inline above
#    (`test_welch_topic_unit_shrinks_n_to_topic_count`,
#     `test_bootstrap_topic_unit_runs_and_brackets_point_estimate`)
# ==========================================================================


# ==========================================================================
# 9. Baseline pools + pool-growth sweep
# ==========================================================================


def test_resolve_pool_removes_target():
    assert resolve_pool(["a", "b", "c", "target"], "target") == ("a", "b", "c")
    assert resolve_pool(["a", "b", "c"], "target") == ("a", "b", "c")


def test_resolve_pool_raises_with_too_few_baselines():
    with pytest.raises(ValueError):
        resolve_pool(["a", "target"], "target")


def test_pool_growth_sweep_null_case_never_rejects():
    """Target's embedding is statistically identical to everyone else's ->
    every subset at every size should fail to reject H0."""
    keys = ("X", "A", "B", "C", "D")
    n = 6
    rng = np.random.default_rng(1)
    raw = 0.1 + 0.02 * rng.standard_normal((5, 5, n))
    d = (raw + raw.transpose(1, 0, 2)) / 2
    for i in range(5):
        d[i, i, :] = 0.0
    dist = EmbeddingSet(keys=keys, distances=d)
    results = pool_growth_sweep(dist, "X", list(keys), sizes=[2, 3, 4])
    for r in results:
        assert r.fraction_rejected == pytest.approx(0.0)
        assert r.n_subsets == r.n_rejected + (r.n_subsets - r.n_rejected)


def test_pool_growth_sweep_separated_case_always_rejects():
    dist = _strongly_separated_universe()
    universe = [k for k in dist.keys if k != "X"]
    results = pool_growth_sweep(dist, "X", universe, sizes=[2, 3, 4])
    for r in results:
        assert r.fraction_rejected == pytest.approx(1.0)


def test_pool_growth_sweep_subset_counts_match_binomial_coefficient():
    from math import comb

    dist = _strongly_separated_universe()
    universe = [k for k in dist.keys if k != "X"]  # 4 candidates
    results = pool_growth_sweep(dist, "X", universe, sizes=[2, 3])
    assert results[0].n_subsets == comb(4, 2)
    assert results[1].n_subsets == comb(4, 3)


def test_pool_growth_sweep_rejects_size_below_two():
    dist = _strongly_separated_universe()
    universe = [k for k in dist.keys if k != "X"]
    with pytest.raises(ValueError):
        pool_growth_sweep(dist, "X", universe, sizes=[1])
