"""Canonical data schema.

Every stage of the pipeline reads and writes one of the three tables defined
here. Legacy on-disk layouts are absorbed by :mod:`parallax_audit.archive`; no
module outside that one should know that the original CSVs ever existed.

The column vocabulary is deliberately small:

``RESPONSES``
    One row per (case study, question, model) -> the model's answer.
``JUDGE_SCORES``
    One row per (case study, question, model, judge, scale) -> ordinal score.
``EMBEDDING_INDEX``
    One row per (case study, question, model) aligned positionally with the
    matching ``.npy`` matrix, so vectors never live inside a dataframe cell.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Identifier columns
# --------------------------------------------------------------------------

CASE_STUDY: Final = "case_study"  # 'cs1' | 'cs2' | 'cs3'
TOPIC: Final = "topic"  # verbatim theme string from the question set
QUESTION_ID: Final = "question_id"  # 'cs1-q007', stable across all stages
QUESTION: Final = "question"
MODEL: Final = "model"  # canonical key, see parallax_audit.registry

# --------------------------------------------------------------------------
# Payload columns
# --------------------------------------------------------------------------

RESPONSE: Final = "response"
JUDGE: Final = "judge"  # canonical judge key, see parallax_audit.registry
SCALE: Final = "scale"  # 2 | 4 | 7 | 10
SCORE: Final = "score"  # ordinal, 1..scale
EXPLANATION: Final = "explanation"
ENCODER: Final = "encoder"  # canonical encoder key
ROW: Final = "row"  # position into the embedding matrix

# --------------------------------------------------------------------------
# Table definitions
# --------------------------------------------------------------------------

RESPONSES: Final = (CASE_STUDY, TOPIC, QUESTION_ID, QUESTION, MODEL, RESPONSE)

JUDGE_SCORES: Final = (
    CASE_STUDY,
    TOPIC,
    QUESTION_ID,
    MODEL,
    JUDGE,
    SCALE,
    SCORE,
    EXPLANATION,
)

EMBEDDING_INDEX: Final = (CASE_STUDY, TOPIC, QUESTION_ID, MODEL, ENCODER, ROW)

VALID_SCALES: Final = (2, 4, 7, 10)

def validate(frame, columns, *, name: str = "frame") -> None:
    """Validate columns, identity keys, metadata consistency, and payload ranges."""
    actual = tuple(frame.columns)
    if actual != tuple(columns):
        missing = [c for c in columns if c not in actual]
        extra = [c for c in actual if c not in columns]
        raise ValueError(
            f"{name} does not match schema.\n"
            f"  expected: {list(columns)}\n"
            f"  actual:   {list(actual)}\n"
            f"  missing:  {missing}\n"
            f"  extra:    {extra}"
        )

    # Null payloads represent failed collection/scoring; identifiers never do.
    import numpy as np
    import pandas as pd

    identifiers = [c for c in (CASE_STUDY, TOPIC, QUESTION_ID, MODEL, JUDGE, ENCODER) if c in actual]
    if any(frame[c].isna().any() or not frame[c].map(lambda v: isinstance(v, str) and bool(v.strip())).all()
           for c in identifiers):
        raise ValueError(f'{name}: identifiers must be nonempty strings')
    keys = [c for c in (CASE_STUDY, QUESTION_ID, MODEL, JUDGE, SCALE, ENCODER) if c in actual]
    if keys and frame.duplicated(keys).any():
        raise ValueError(f'{name}: duplicate row keys {keys}')
    question_keys = [c for c in (CASE_STUDY, QUESTION_ID) if c in actual]
    for c in (TOPIC, QUESTION):
        if c in actual and question_keys and (frame.groupby(question_keys)[c].nunique(dropna=False) > 1).any():
            raise ValueError(f'{name}: inconsistent {c} for the same question')
    if SCALE in actual:
        if not frame[SCALE].isin(VALID_SCALES).all():
            raise ValueError(f'{name}: unsupported ordinal scale')
        scores = pd.to_numeric(frame[SCORE], errors='raise')
        present = scores.notna()
        if ((scores[present] < 1) | (scores[present] > frame.loc[present, SCALE]) |
                (scores[present] % 1 != 0) | ~np.isfinite(scores[present])).any():
            raise ValueError(f'{name}: scores must be integers within their scale')
    if ROW in actual:
        positions = pd.to_numeric(frame[ROW], errors='raise')
        if positions.isna().any() or (positions % 1 != 0).any() or (positions < 0).any() or positions.duplicated().any():
            raise ValueError(f'{name}: embedding row positions must be unique nonnegative integers')
