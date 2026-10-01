"""Orchestration layer: canonical tables in, statistical results out.

This is the seam between two deliberately separate halves of the package.
``parallax_audit.stats`` is pure: it takes numpy arrays and mappings, does no I/O, and
knows nothing about case studies or encoders, which is what makes every
estimator unit-testable against hand-computed fixtures. ``parallax_audit.data`` is the
opposite: it knows only about loading tables. Neither should have to import the
other, so the work of "load the right slice, reshape it, call the right test"
lives here, and the CLI stays a presentation layer over these functions.

Each public function returns a tidy ``DataFrame``, one row per test, so results
compose directly into the LaTeX tables emitted by ``parallax_audit.report``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Literal, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from parallax_audit import registry, schema, stats

logger = logging.getLogger(__name__)

Unit = Literal["question", "topic"]


# --------------------------------------------------------------------------
# Reshaping canonical tables into the array forms parallax_audit.stats expects
# --------------------------------------------------------------------------


def embedding_matrices(
    encoder: str, case_study: str, *, models: Sequence[str] | None = None, outputs: Path = Path("outputs")
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Load embeddings as ``{model: (N, dim)}`` aligned on a common question set.

    Returns ``(matrices, question_ids, topics)``. Models are aligned by
    ``question_id`` and restricted to the questions every model covers, so a
    gap in one model's coverage (the archive has one -- see
    the archived CS1 Instructor vectors) shrinks the shared set rather than silently
    misaligning rows.
    """
    from parallax_audit.data import load_embeddings

    index, matrix = load_embeddings(encoder, case_study, outputs=outputs)
    if models is not None:
        missing = set(models) - set(index[schema.MODEL])
        if missing:
            raise ValueError(f"missing required models: {sorted(missing)}")
        index = index[index[schema.MODEL].isin(models)]

    per_model = {
        model: frame.set_index(schema.QUESTION_ID)
        for model, frame in index.groupby(schema.MODEL, sort=True)
    }
    if not per_model:
        raise ValueError(f"no embeddings found for {encoder}/{case_study}")

    shared = sorted(set.intersection(*(set(f.index) for f in per_model.values())))
    if not shared:
        raise ValueError(f"models share no questions for {encoder}/{case_study}")

    dropped = {m: len(f) - len(shared) for m, f in per_model.items() if len(f) > len(shared)}
    if dropped:
        logger.warning(
            "%s/%s: restricted to %d shared questions; dropped per model: %s",
            encoder,
            case_study,
            len(shared),
            dropped,
        )

    matrices = {
        model: matrix[frame.loc[shared, schema.ROW].to_numpy()]
        for model, frame in per_model.items()
    }
    first = next(iter(per_model.values()))
    topics = first.loc[shared, schema.TOPIC].tolist()
    return matrices, shared, topics


def judge_matrices(
    case_study: str, judge: str, scale: int, *, models: Sequence[str] | None = None, outputs: Path = Path("outputs")
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Load judge scores as ``{model: (N,)}`` aligned on a common question set.

    Returns ``(scores, question_ids, topics)``.
    """
    from parallax_audit.data import load_judge_scores

    frame = load_judge_scores(case_study=case_study, judge=judge, scale=scale, outputs=outputs)
    if models is not None:
        missing = set(models) - set(frame[schema.MODEL])
        if missing:
            raise ValueError(f"missing required models: {sorted(missing)}")
        frame = frame[frame[schema.MODEL].isin(models)]
    if frame.empty:
        raise ValueError(f"no {judge} scores at scale {scale} for {case_study}")

    wide = frame.pivot(
        index=schema.QUESTION_ID, columns=schema.MODEL, values=schema.SCORE
    ).sort_index()
    complete = wide.dropna(axis=0, how="any")
    if len(complete) != len(wide):
        logger.warning("%s/%s: excluded %d incompletely scored questions", case_study, judge, len(wide) - len(complete))
    wide = complete
    if wide.empty:
        raise ValueError(
            f"no question is scored for every model in {case_study}/{judge}/scale{scale}"
        )

    topic_of = (
        frame.drop_duplicates(schema.QUESTION_ID)
        .set_index(schema.QUESTION_ID)[schema.TOPIC]
        .to_dict()
    )
    scores = {model: wide[model].to_numpy(dtype=float) for model in wide.columns}
    question_ids = wide.index.tolist()
    return scores, question_ids, [topic_of[q] for q in question_ids]


def _members(target: str, pool: str | None) -> list[str]:
    """Resolve a fixed experiment before intersecting question coverage."""
    candidates = registry.pool(pool).members if pool else registry.original_eight()
    return [target, *stats.resolve_pool(candidates, target)]


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------


def _row(result, **extra) -> dict:
    return {**extra, **dataclasses.asdict(result)}


def embedding_analysis(
    case_study: str,
    encoder: str,
    *,
    pool: str | None = None,
    unit: Unit = "question",
    permute: bool = False,
    outputs: Path = Path("outputs"),
) -> pd.DataFrame:
    """One-sided Welch test on embedding deviation (paper Table 1 / Table 5).

    With ``permute``, every candidate is treated as the target in turn and the
    full ablation is returned, ordered by ascending p-value.
    """
    cs = registry.case_study(case_study)
    members = _members(cs.target, pool)
    matrices, _, topics = embedding_matrices(encoder, case_study, models=members, outputs=outputs)
    dist = stats.build_distance_tensor(matrices)
    topic_arg = topics if unit == "topic" else None

    if permute:
        table = stats.permute_embedding_targets(
            dist, list(matrices), unit=unit, topics=topic_arg
        )
        return table.assign(case_study=case_study, encoder=encoder, pool=pool or "all")

    result = stats.embedding_welch_test(
        dist, cs.target, members[1:], unit=unit, topics=topic_arg
    )
    return pd.DataFrame(
        [_row(result, case_study=case_study, encoder=encoder, pool=pool or "all")]
    )


def judge_analysis(
    case_study: str,
    judge: str,
    scale: int,
    *,
    pool: str | None = None,
    unit: Unit = "question",
    permute: bool = False,
    B: int = 20_000,
    seed: int = 0,
    outputs: Path = Path("outputs"),
) -> pd.DataFrame:
    """Bootstrap test on judge deviation (paper Table 3 / Table 6)."""
    cs = registry.case_study(case_study)
    members = _members(cs.target, pool)
    scores, _, topics = judge_matrices(case_study, judge, scale, models=members, outputs=outputs)
    topic_arg = topics if unit == "topic" else None

    if permute:
        table = stats.permute_judge_targets(
            scores, list(scores), unit=unit, topics=topic_arg, B=B, seed=seed
        )
        return table.assign(
            case_study=case_study, judge=judge, scale=scale, pool=pool or "all"
        )

    result = stats.judge_bootstrap_test(
        scores, cs.target, members[1:], unit=unit, topics=topic_arg, B=B, seed=seed
    )
    return pd.DataFrame(
        [
            _row(
                result,
                case_study=case_study,
                judge=judge,
                scale=scale,
                pool=pool or "all",
            )
        ]
    )


def agreement_analysis(
    case_study: str, scale: int, judge_a: str, judge_b: str, *, outputs: Path = Path("outputs")
) -> pd.DataFrame:
    """Inter-judge agreement on raw scores (paper Table 2).

    Compared over the (model, question) pairs both judges scored, so a partial
    judge run degrades the sample size rather than the alignment.
    """
    from parallax_audit.data import load_judge_scores

    keys = [schema.CASE_STUDY, schema.QUESTION_ID, schema.MODEL]
    frames = []
    for judge in (judge_a, judge_b):
        frame = load_judge_scores(case_study=case_study, judge=judge, scale=scale, outputs=outputs)
        if frame.duplicated(keys).any():
            raise ValueError(f"duplicate scored items for {judge}")
        frames.append(frame[keys + [schema.SCORE]].dropna(subset=[schema.SCORE]))
    shared = frames[0].merge(frames[1], on=keys, how="inner", validate="one_to_one",
                             suffixes=("_a", "_b"))
    if shared.empty:
        raise ValueError(f"{judge_a} and {judge_b} share no scored items for {case_study}")
    models = shared[schema.MODEL].unique()
    result = stats.inter_judge_agreement(
        shared[f"{schema.SCORE}_a"].to_numpy(),
        shared[f"{schema.SCORE}_b"].to_numpy(),
        scale=scale,
    )
    return pd.DataFrame(
        [
            _row(
                result,
                case_study=case_study,
                scale=scale,
                judge_a=judge_a,
                judge_b=judge_b,
                n_models=len(models),
            )
        ]
    )


def correlation_analysis(
    case_study: str,
    encoder: str,
    judge: str,
    scale: int,
    *,
    judge_value: Literal["raw", "deviation"] = "raw",
    outputs: Path = Path("outputs"),
) -> pd.DataFrame:
    """Cross-method Spearman correlation for the target (rebuttal).

    ``judge_value='raw'`` reproduces the published values; ``'deviation'`` uses
    the leave-one-out judge deviation instead. See the paper's method description.
    """
    cs = registry.case_study(case_study)
    members = _members(cs.target, None)
    matrices, emb_questions, _ = embedding_matrices(encoder, case_study, models=members, outputs=outputs)
    scores, judge_questions, _ = judge_matrices(case_study, judge, scale, models=members, outputs=outputs)

    shared = [q for q in emb_questions if q in set(judge_questions)]
    if not shared:
        raise ValueError(f"embedding and judge question sets disjoint for {case_study}")
    emb_pos = {q: i for i, q in enumerate(emb_questions)}
    judge_pos = {q: i for i, q in enumerate(judge_questions)}
    take_emb = [emb_pos[q] for q in shared]
    take_judge = [judge_pos[q] for q in shared]

    dist = stats.build_distance_tensor(matrices)
    deviation = stats.deviation_vector(dist, cs.target, members[1:])[take_emb]

    peers = (
        {m: scores[m][take_judge] for m in scores if m != cs.target}
        if judge_value == "deviation"
        else None
    )
    result = stats.spearman_cross_method(
        deviation,
        scores[cs.target][take_judge],
        peer_scores=peers,
        judge_value=judge_value,
        case_study=case_study,
        judge=judge,
        encoder=encoder,
        scale=scale,
    )
    return pd.DataFrame([_row(result)])


def pool_growth_analysis(*, outputs: Path = Path("outputs")) -> pd.DataFrame:
    """Exhaustive pool-growth sweep on the null control (rebuttal).

    Reads its whole configuration from the ``pool_growth`` block of
    ``config/pools.yaml``: the false-positive rate is only interpretable
    against a case study where no proprietary alignment is hypothesised.
    """
    config = registry.pool_growth()
    case_study = config["case_study"]
    members = [config["target"], *config["universe"]]
    matrices, _, _ = embedding_matrices("instructor", case_study, models=members, outputs=outputs)

    universe = [m for m in config["universe"] if m in matrices]
    absent = [m for m in config["universe"] if m not in matrices]
    if absent:
        logger.warning("pool-growth sweep: no data for %s; excluded", absent)

    dist = stats.build_distance_tensor(matrices)
    results = stats.pool_growth_sweep(
        dist,
        config["target"],
        universe,
        [k for k in config["sizes"] if k <= len(universe)],
    )
    return pd.DataFrame([_row(r, case_study=case_study) for r in results])
