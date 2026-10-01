"""Legacy corpus parsing -- the only module that knows the original on-disk
layout under ``datasets/``.

Every function here is pure: given the case study's canonical question set
(see :mod:`parallax_audit.data`) and the corpus root, it returns one
schema-conforming table (or, for embeddings, a table plus its aligned
matrix). None of them write anything -- that is :mod:`parallax_audit.data`'s
job -- and none of them guess: an unresolvable filename, a row count that
does not match the canonical question set, or an out-of-range score raises
immediately rather than silently dropping or truncating.

The corpus never agrees with itself on spelling: response columns are
``Claude_Response`` in one file and ``DeepSeekAWS_Responses`` in another,
judge directories are ``deepSeekAWS`` in one case study and ``deepseekAWS``
in the next, and the two embedding encoders use ``.json`` and ``.jsonl`` for
the same JSON-lines format. All of that is absorbed here via
:func:`parallax_audit.registry.resolve_model`/:func:`parallax_audit.registry.resolve_encoder`
and substring column matching, so nothing downstream needs to know it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from parallax_audit import registry, schema

logger = logging.getLogger(__name__)

#: Canonical label for the archived GPT judge scores. The exact historical
#: deployment is unconfirmed. No Gemini scores are archived.
LEGACY_JUDGE = "gpt52"

#: Legacy raw/judged CSVs always spell this column exactly this way; only the
#: per-model response/score/explanation columns vary.
QUESTION_COL = "Question"

#: Fine-tuning artifacts (the Llama3-8B sweep) live alongside the eight
#: canonical models' files in ``raw_responses/`` and ``embeddings/`` and must
#: never be mistaken for one of them.
_FINE_TUNE_ARTIFACT = re.compile(r"^llama3", re.IGNORECASE)

#: Matches judge filenames case-insensitively regardless of the
#: ``judged``/``Judged``/absent infix (``MistralLarge_CS1_Judged_scale4.csv``,
#: ``MistralLarge_CS2_scale10.csv``, ``deepseekAWS_CS2_responses_judged_scale10.csv``).
_SCALE_FILE = re.compile(r"cale(?P<scale>\d+)\.csv$", re.IGNORECASE)

#: Pre-concatenated ("all_models_with_embeddings.*") and fine-tuning
#: ("llama3*") artifacts under ``embeddings/``; ``embedding_CS1_df.csv`` is
#: excluded by extension alone (see ``_EMBED_EXTS``).
_EMBED_EXCLUDE_PREFIX = re.compile(r"^(all_models_with_embeddings|llama3)", re.IGNORECASE)
_EMBED_EXTS = (".json", ".jsonl")


def _find_column(columns: pd.Index, needle_parts: tuple[str, ...]) -> str:
    """Return the one column whose lowercased name contains every part.

    The corpus never agrees on a column's exact spelling (``Bias Score`` vs
    ``Bias Score (Judged by GPT4o)``, ``Explanation-GPT`` vs
    ``Explanation-GPT (scale=2)``) but always agrees on its meaning, so
    substring matching is the only rule stable enough to hardcode. Raising on
    zero or multiple matches turns an unexpected header into a loud failure
    instead of a silently wrong column.
    """
    matches = [c for c in columns if all(part in c.lower() for part in needle_parts)]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one column matching {needle_parts}, found {matches} in {list(columns)}"
        )
    return matches[0]


# --------------------------------------------------------------------------
# Raw responses
# --------------------------------------------------------------------------


def candidate_raw_files(cs: registry.CaseStudy, root: Path) -> list[Path]:
    """Top-level response CSVs for one case study, sorted by filename.

    Non-recursive; fine-tuning (``llama3*``) files are excluded explicitly.
    """
    directory = root / "raw_responses" / cs.legacy_dir
    files = sorted(p for p in directory.glob("*.csv") if not _FINE_TUNE_ARTIFACT.match(p.stem))
    if not files:
        raise FileNotFoundError(f"no raw response CSVs under {directory}")
    return files


def _raw_files_by_model(cs: registry.CaseStudy, root: Path) -> dict[str, Path]:
    """Canonical model key -> its raw response CSV, asserting full coverage."""
    mapping: dict[str, Path] = {}
    for path in candidate_raw_files(cs, root):
        model_key = registry.resolve_model(path.name)
        if model_key in mapping:
            raise ValueError(
                f"{cs.key}: both {mapping[model_key].name!r} and {path.name!r} "
                f"resolve to model {model_key!r}"
            )
        mapping[model_key] = path

    expected = set(registry.original_eight())
    if set(mapping) != expected:
        missing = expected - set(mapping)
        extra = set(mapping) - expected
        raise ValueError(f"{cs.key}: raw responses missing={sorted(missing)} extra={sorted(extra)}")
    return mapping


def parse_responses(questions: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Parse all eight models' raw response CSVs for one case study.

    ``questions`` is that case study's canonical question set; joining each
    file's verbatim ``Question`` text against it is what assigns the stable
    ``question_id`` -- a raw file's own row order is never trusted, only the
    reference file's (baked into ``questions``) is.
    """
    cs_key = str(questions[schema.CASE_STUDY].iloc[0])
    cs = registry.case_study(cs_key)
    expected_n = cs.n_questions
    qid_of = questions.set_index(schema.QUESTION)[schema.QUESTION_ID]
    topic_of = questions.set_index(schema.QUESTION_ID)[schema.TOPIC]

    frames: list[pd.DataFrame] = []
    for model_key, path in sorted(_raw_files_by_model(cs, root).items()):
        raw = pd.read_csv(path)
        if len(raw) != expected_n:
            raise ValueError(f"{path}: expected {expected_n} rows, got {len(raw)}")
        response_col = _find_column(raw.columns, ("response",))

        unknown = set(raw[QUESTION_COL]) - set(qid_of.index)
        if unknown:
            raise ValueError(f"{path}: {len(unknown)} question(s) not in the canonical set: {sorted(unknown)[:3]}")
        qid = raw[QUESTION_COL].map(qid_of)

        frames.append(
            pd.DataFrame(
                {
                    schema.CASE_STUDY: cs_key,
                    schema.TOPIC: qid.map(topic_of).to_numpy(),
                    schema.QUESTION_ID: qid.to_numpy(),
                    schema.QUESTION: raw[QUESTION_COL].to_numpy(),
                    schema.MODEL: model_key,
                    schema.RESPONSE: raw[response_col].to_numpy(),
                }
            )
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values([schema.QUESTION_ID, schema.MODEL]).reset_index(drop=True)
    result = result[list(schema.RESPONSES)]
    schema.validate(result, schema.RESPONSES, name=f"{cs_key} responses")
    return result


# --------------------------------------------------------------------------
# Judge scores
# --------------------------------------------------------------------------


def parse_judge_scores(questions: pd.DataFrame, root: Path) -> dict[int, pd.DataFrame]:
    """Parse all eight models' judge CSVs for one case study, keyed by scale.

    Directory names under ``judged_responses/<case study>/`` are as
    inconsistent as the model aliases they are supposed to spell
    (``deepSeekAWS`` vs ``deepseekAWS``), so the model is resolved from the
    file's *basename* only, via a recursive, extension-anchored,
    case-insensitive glob on ``*cale{N}.csv``.
    """
    cs_key = str(questions[schema.CASE_STUDY].iloc[0])
    cs = registry.case_study(cs_key)
    expected_n = cs.n_questions
    qid_of = questions.set_index(schema.QUESTION)[schema.QUESTION_ID]
    topic_of = questions.set_index(schema.QUESTION_ID)[schema.TOPIC]

    directory = root / "judged_responses" / cs.legacy_dir
    by_scale: dict[int, list[pd.DataFrame]] = {scale: [] for scale in schema.VALID_SCALES}
    seen_models: dict[int, set[str]] = {scale: set() for scale in schema.VALID_SCALES}

    for path in sorted(directory.rglob("*.csv")):
        match = _SCALE_FILE.search(path.name)
        if not match:
            continue
        scale = int(match.group("scale"))
        if scale not in schema.VALID_SCALES:
            raise ValueError(f"{path}: scale {scale} is not one of {schema.VALID_SCALES}")

        model_key = registry.resolve_model(path.name)
        if model_key in seen_models[scale]:
            raise ValueError(f"{cs_key} scale={scale}: duplicate judge file for model {model_key!r} ({path})")
        seen_models[scale].add(model_key)

        raw = pd.read_csv(path)
        if len(raw) != expected_n:
            raise ValueError(f"{path}: expected {expected_n} rows, got {len(raw)}")
        score_col = _find_column(raw.columns, ("bias", "score"))
        explanation_col = _find_column(raw.columns, ("explanation",))

        unknown = set(raw[QUESTION_COL]) - set(qid_of.index)
        if unknown:
            raise ValueError(f"{path}: {len(unknown)} question(s) not in the canonical set")
        qid = raw[QUESTION_COL].map(qid_of)

        by_scale[scale].append(
            pd.DataFrame(
                {
                    schema.CASE_STUDY: cs_key,
                    schema.TOPIC: qid.map(topic_of).to_numpy(),
                    schema.QUESTION_ID: qid.to_numpy(),
                    schema.MODEL: model_key,
                    schema.JUDGE: LEGACY_JUDGE,
                    schema.SCALE: scale,
                    schema.SCORE: raw[score_col].to_numpy(),
                    schema.EXPLANATION: raw[explanation_col].to_numpy(),
                }
            )
        )

    expected_models = set(registry.original_eight())
    result: dict[int, pd.DataFrame] = {}
    for scale in schema.VALID_SCALES:
        if seen_models[scale] != expected_models:
            missing = expected_models - seen_models[scale]
            extra = seen_models[scale] - expected_models
            raise ValueError(f"{cs_key} scale={scale}: judge files missing={sorted(missing)} extra={sorted(extra)}")

        frame = pd.concat(by_scale[scale], ignore_index=True)
        frame = frame.sort_values([schema.QUESTION_ID, schema.MODEL]).reset_index(drop=True)
        frame = frame[list(schema.JUDGE_SCORES)]

        if frame[schema.SCORE].isna().any():
            raise ValueError(f"{cs_key} scale={scale}: null score(s) in judge data")
        out_of_range = ~frame[schema.SCORE].between(1, scale)
        if out_of_range.any():
            bad = frame.loc[out_of_range, schema.SCORE].tolist()
            raise ValueError(f"{cs_key} scale={scale}: score(s) outside 1..{scale}: {bad[:5]}")

        schema.validate(frame, schema.JUDGE_SCORES, name=f"{cs_key} judge scores scale={scale}")
        result[scale] = frame

    return result


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def discover_embedding_dirs(root: Path) -> dict[str, Path]:
    """Canonical encoder key -> its legacy directory (``instructor`` -> ``.../INSTRUCTOR``)."""
    base = root / "embeddings"
    return {registry.resolve_encoder(d.name): d for d in sorted(base.iterdir()) if d.is_dir()}


def _candidate_embedding_files(cs_dir: Path) -> list[Path]:
    """Model embedding files in one (encoder, case study) directory.

    Excludes the pre-concatenated ``all_models_with_embeddings.*``, the
    fine-tuning ``llama3*`` sweep, and anything that is not ``.json``/
    ``.jsonl`` -- which also drops ``embedding_CS1_df.csv`` and stray
    ``.DS_Store`` files for free.
    """
    return sorted(
        p
        for p in cs_dir.iterdir()
        if p.suffix.lower() in _EMBED_EXTS and not _EMBED_EXCLUDE_PREFIX.match(p.stem)
    )


def _embedding_files_by_model(cs: registry.CaseStudy, cs_dir: Path) -> dict[str, Path]:
    """Canonical model key -> its embedding file, deduplicating stray copies.

    ``embeddings/INSTRUCTOR/CaseStudy2_US/claude_with_embeddings.json`` is a
    duplicate of the correctly-suffixed ``claude_with_embeddings_CS2.json``
    sitting right next to it. When a model resolves to more than one file,
    the one whose stem carries the case study's short code (``CS2``) wins and
    the rest are logged as excluded duplicates -- never silently summed.
    """
    by_model: dict[str, list[Path]] = {}
    for path in _candidate_embedding_files(cs_dir):
        by_model.setdefault(registry.resolve_model(path.name), []).append(path)

    resolved: dict[str, Path] = {}
    for model_key, paths in by_model.items():
        if len(paths) == 1:
            resolved[model_key] = paths[0]
            continue

        suffixed = [p for p in paths if cs.short.lower() in p.stem.lower()]
        if len(suffixed) != 1:
            raise ValueError(
                f"{cs.key}/{model_key}: ambiguous embedding files {[p.name for p in paths]}; "
                f"expected exactly one with the {cs.short!r} suffix"
            )
        dropped = [p for p in paths if p != suffixed[0]]
        logger.warning(
            "%s/%s: excluding stray duplicate embedding file(s) %s (keeping %s)",
            cs.key,
            model_key,
            [p.name for p in dropped],
            suffixed[0].name,
        )
        resolved[model_key] = suffixed[0]

    expected = set(registry.original_eight())
    if set(resolved) != expected:
        missing = expected - set(resolved)
        extra = set(resolved) - expected
        raise ValueError(f"{cs.key}: embedding files missing={sorted(missing)} extra={sorted(extra)}")
    return resolved


def parse_embeddings(questions: pd.DataFrame, root: Path, encoder_key: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Parse one encoder's embedding files for one case study.

    Returns ``(index_df, matrix)`` positionally aligned and sorted by
    ``(question_id, model)``, matching the order :mod:`parallax_audit.cli`'s
    ``embed`` subcommand produces for a freshly-computed encoder.

    A model file short of the full question count (e.g. one dropped
    response) is *not* an error: it is logged as a structured warning naming
    the missing ``question_id`` and the resulting table is simply shorter for
    that model -- the alternative, silently padding or dropping to force a
    square table, is exactly the kind of ragged-data bug this loader exists
    to catch.
    """
    cs_key = str(questions[schema.CASE_STUDY].iloc[0])
    cs = registry.case_study(cs_key)
    encoder = registry.encoder(encoder_key)

    encoder_dirs = discover_embedding_dirs(root)
    if encoder_key not in encoder_dirs:
        raise FileNotFoundError(f"no legacy embedding directory for encoder {encoder_key!r} under {root}")
    cs_dir = encoder_dirs[encoder_key] / cs.legacy_dir

    qid_of = questions.set_index(schema.QUESTION)[schema.QUESTION_ID]
    topic_of = questions.set_index(schema.QUESTION_ID)[schema.TOPIC]
    all_question_ids = set(questions[schema.QUESTION_ID])

    index_frames: list[pd.DataFrame] = []
    vectors: list[np.ndarray] = []
    for model_key, path in sorted(_embedding_files_by_model(cs, cs_dir).items()):
        raw = pd.read_json(path, lines=True)  # JSON-lines under both .json and .jsonl

        unknown = set(raw[QUESTION_COL]) - set(qid_of.index)
        if unknown:
            raise ValueError(f"{path}: {len(unknown)} question(s) not in the canonical set")
        qid = raw[QUESTION_COL].map(qid_of)

        missing_ids = sorted(all_question_ids - set(qid))
        if missing_ids:
            logger.warning(
                "%s/%s/%s: missing %d/%d response embedding(s); question_id(s) %s absent from %s",
                cs.key,
                model_key,
                encoder_key,
                len(missing_ids),
                len(all_question_ids),
                missing_ids,
                path.name,
            )

        index_frames.append(
            pd.DataFrame(
                {
                    schema.CASE_STUDY: cs_key,
                    schema.TOPIC: qid.map(topic_of).to_numpy(),
                    schema.QUESTION_ID: qid.to_numpy(),
                    schema.MODEL: model_key,
                    schema.ENCODER: encoder_key,
                }
            )
        )
        vectors.append(np.stack(raw["embedding"].to_numpy()).astype(np.float32, copy=False))

    index_df = pd.concat(index_frames, ignore_index=True)
    matrix = np.concatenate(vectors, axis=0)
    if matrix.shape[1] != encoder.dim:
        raise ValueError(f"{cs.key}/{encoder_key}: embedding dim {matrix.shape[1]} != registry dim {encoder.dim}")

    order = index_df.sort_values([schema.QUESTION_ID, schema.MODEL]).index.to_numpy()
    index_df = index_df.loc[order].reset_index(drop=True)
    matrix = matrix[order]
    index_df[schema.ROW] = np.arange(len(index_df))
    index_df = index_df[list(schema.EMBEDDING_INDEX)]

    schema.validate(index_df, schema.EMBEDDING_INDEX, name=f"{cs_key} {encoder_key} embedding index")
    return index_df, matrix
