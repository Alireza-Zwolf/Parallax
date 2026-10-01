"""Canonical artifact I/O, frozen questions, and archived-corpus ingestion."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

import numpy as np
import pandas as pd

from parallax_audit import archive as legacy, registry, schema

DEFAULT_ROOT: Final = Path('datasets')
DEFAULT_OUTPUTS: Final = Path('outputs')


# Atomic artifact formats
def _atomic(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as file:
        temporary = Path(file.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write real Parquet; never disguise another format under this extension."""
    try:
        _atomic(path, lambda temporary: frame.to_parquet(temporary, index=False, engine='pyarrow'))
    except ImportError as exc:
        raise RuntimeError('Parquet requires pyarrow; install the project dependencies') from exc


def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine='pyarrow')
    except ImportError as exc:
        raise RuntimeError('Parquet requires pyarrow; install the project dependencies') from exc


def validate_matrix(matrix: np.ndarray, *, dim: int | None = None) -> None:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError('embedding matrix must be nonempty and two-dimensional')
    if dim is not None and matrix.shape[1] != dim:
        raise ValueError(f'embedding dimension {matrix.shape[1]} != expected {dim}')
    if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
        raise ValueError('embeddings must be finite, nonzero vectors')


def write_matrix(matrix: np.ndarray, path: Path) -> None:
    matrix = np.asarray(matrix, dtype=np.float32)
    validate_matrix(matrix)
    def write(temporary):
        with temporary.open('wb') as file:
            np.save(file, matrix, allow_pickle=False)
    _atomic(path, write)


def read_matrix(path: Path, *, dim: int | None = None) -> np.ndarray:
    matrix = np.load(path, allow_pickle=False)
    validate_matrix(matrix, dim=dim)
    return matrix


def write_json(value, path: Path) -> None:
    _atomic(path, lambda temporary: temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n'))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic(path, lambda temporary: frame.to_csv(temporary, index=False))


def write_text(text: str, path: Path) -> None:
    _atomic(path, lambda temporary: temporary.write_text(text))


# Frozen question set and canonical readers
QUESTION_SET = (schema.CASE_STUDY, schema.TOPIC, schema.QUESTION_ID, schema.QUESTION)


def build_question_set(case_study: str) -> pd.DataFrame:
    """Read the frozen paper question set."""
    spec = registry.case_study(case_study)
    frame = pd.read_csv(registry.CONFIG_DIR / 'questions' / f'{case_study}.csv')
    schema.validate(frame, QUESTION_SET, name=f'{case_study} questions')
    if len(frame) != spec.n_questions or set(frame[schema.CASE_STUDY]) != {case_study}:
        raise ValueError(f'{case_study}: question snapshot does not match configuration')
    if frame[schema.QUESTION].duplicated().any():
        raise ValueError(f'{case_study}: duplicate question text')
    return frame


_JUDGE_FILENAME = re.compile(r"^(?P<judge>.+)_scale(?P<scale>\d+)\.parquet$")


def _case_study_keys(case_study: str | None) -> list[str]:
    if case_study is not None:
        registry.case_study(case_study)  # validates; raises KeyError with known keys
        return [case_study]
    return sorted(registry.case_studies())


def load_questions(case_study: str | None = None, *, outputs: Path = DEFAULT_OUTPUTS) -> pd.DataFrame:
    """Read the canonical question set(s) written by :func:`parallax_audit.data.run_ingest`."""
    frames = [read_table(outputs / "questions" / f"{cs}.parquet") for cs in _case_study_keys(case_study)]
    frame = pd.concat(frames, ignore_index=True)
    schema.validate(frame, QUESTION_SET, name="questions")
    return frame


def load_responses(case_study: str | None = None, *, outputs: Path = DEFAULT_OUTPUTS) -> pd.DataFrame:
    """Read every model's responses for one or all case studies.

    One Parquet file per (case study, model) on disk (see ``parallax_audit.cli``'s
    ``query`` subcommand), so a model queried after ingest is picked up the
    same way as one ingested from the legacy corpus.
    """
    frames = []
    for cs in _case_study_keys(case_study):
        cs_dir = outputs / "responses" / cs
        paths = sorted(cs_dir.glob("*.parquet")) if cs_dir.exists() else []
        if not paths:
            raise FileNotFoundError(f"no responses under {cs_dir}; run `parallax_audit ingest` first")
        frames.extend(read_table(path) for path in paths)

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values([schema.CASE_STUDY, schema.QUESTION_ID, schema.MODEL]).reset_index(drop=True)
    schema.validate(frame, schema.RESPONSES, name="responses")
    return frame


def _parse_judge_filename(path: Path) -> tuple[str, int]:
    match = _JUDGE_FILENAME.match(path.name)
    if not match:
        raise ValueError(f"unexpected judge score filename: {path.name}")
    return match.group("judge"), int(match.group("scale"))


def load_judge_scores(
    case_study: str | None = None,
    judge: str | None = None,
    scale: int | None = None,
    *,
    outputs: Path = DEFAULT_OUTPUTS,
) -> pd.DataFrame:
    """Read judge scores, optionally filtered to one judge and/or one ordinal scale.

    One Parquet file per (case study, judge, scale) on disk (see
    ``parallax_audit.cli``'s ``judge`` subcommand).
    """
    if judge is not None:
        registry.judge(judge)  # validates
    if scale is not None and scale not in schema.VALID_SCALES:
        raise ValueError(f"unknown scale {scale!r}; known: {list(schema.VALID_SCALES)}")

    frames = []
    for cs in _case_study_keys(case_study):
        cs_dir = outputs / "judge_scores" / cs
        if not cs_dir.exists():
            raise FileNotFoundError(f"no judge scores under {cs_dir}; run `parallax_audit ingest` first")
        for path in sorted(cs_dir.glob("*_scale*.parquet")):
            file_judge, file_scale = _parse_judge_filename(path)
            if judge is not None and file_judge != judge:
                continue
            if scale is not None and file_scale != scale:
                continue
            frames.append(read_table(path))

    if not frames:
        raise FileNotFoundError(
            f"no judge score files matched case_study={case_study!r} judge={judge!r} scale={scale!r}"
        )

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values([schema.CASE_STUDY, schema.QUESTION_ID, schema.MODEL]).reset_index(drop=True)
    schema.validate(frame, schema.JUDGE_SCORES, name="judge_scores")
    return frame


def load_embeddings(
    encoder: str, case_study: str, *, outputs: Path = DEFAULT_OUTPUTS
) -> tuple[pd.DataFrame, np.ndarray]:
    """Read one (encoder, case study) pair's embedding index and matrix.

    Returns ``(index_df, matrix)`` where ``index_df.row`` is a 0..n-1
    permutation giving each row's position in ``matrix`` -- the two must
    never be re-sorted independently.
    """
    dim = registry.encoder(encoder).dim  # validates
    registry.case_study(case_study)  # validates

    base = outputs / "embeddings" / case_study / encoder
    index_path = base.with_suffix(".index.parquet")
    matrix_path = base.with_suffix(".npy")
    if not index_path.exists() or not matrix_path.exists():
        raise FileNotFoundError(
            f"no embeddings for encoder={encoder!r} case_study={case_study!r} under {base.parent}; "
            "run `parallax_audit ingest` or `parallax_audit embed` first"
        )

    index_df = read_table(index_path)
    matrix = read_matrix(matrix_path, dim=dim).astype(np.float32, copy=False)
    schema.validate(index_df, schema.EMBEDDING_INDEX, name=f"{case_study}/{encoder} embedding index")

    if len(index_df) != matrix.shape[0]:
        raise ValueError(f"{index_path}: {len(index_df)} row(s) but {matrix_path} has {matrix.shape[0]}")
    if not np.array_equal(index_df[schema.ROW].to_numpy(), np.arange(len(index_df))):
        raise ValueError(f"{index_path}: {schema.ROW!r} is not a 0..n-1 permutation aligned to the matrix")

    return index_df, matrix


# Archived-corpus ingestion
@dataclass
class IngestReport:
    """Counts and defects from one :func:`run_ingest` call.

    Kept as data rather than only logged: a caller (or a test) should be able
    to assert "the 99-row defect was found" without grepping log output.
    """

    n_questions: dict[str, int] = field(default_factory=dict)
    n_responses: dict[str, int] = field(default_factory=dict)
    n_judge_scores: dict[tuple[str, int], int] = field(default_factory=dict)
    n_embeddings: dict[tuple[str, str], int] = field(default_factory=dict)
    defects: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["ingest summary", "  questions:"]
        lines += [f"    {cs}: {n}" for cs, n in sorted(self.n_questions.items())]
        lines.append("  responses:")
        lines += [f"    {cs}: {n}" for cs, n in sorted(self.n_responses.items())]
        lines.append("  judge_scores:")
        lines += [f"    {cs} scale={scale}: {n}" for (cs, scale), n in sorted(self.n_judge_scores.items())]
        lines.append("  embeddings:")
        lines += [f"    {cs}/{enc}: {n}" for (cs, enc), n in sorted(self.n_embeddings.items())]
        lines.append(f"  defects: {len(self.defects)}")
        lines += [f"    - {defect}" for defect in self.defects]
        return "\n".join(lines)


class _DefectCollector(logging.Handler):
    """Mirrors WARNING+ records from :mod:`parallax_audit.archive` into the report.

    The archive parser logs every defect it finds (a dropped row, an
    excluded duplicate file); this handler is the only thing that lets
    :func:`run_ingest`'s return value be self-contained, so a caller does not
    have to also configure logging to know the corpus was not clean.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def run_ingest(*, root: Path = DEFAULT_ROOT, outputs: Path = DEFAULT_OUTPUTS) -> IngestReport:
    """Parse the legacy corpus and (re)write every canonical table under ``outputs``.

    Idempotent: every write goes through :func:`parallax_audit.data.write_table`
    / :func:`~parallax_audit.data.write_matrix`, both full overwrites, so
    re-running this is always safe.
    """
    legacy_logger = logging.getLogger(legacy.__name__)
    collector = _DefectCollector()
    legacy_logger.addHandler(collector)
    try:
        report = IngestReport()
        cs_keys = sorted(registry.case_studies())
        question_sets = {cs_key: build_question_set(cs_key) for cs_key in cs_keys}

        for cs_key, frame in question_sets.items():
            write_table(frame, outputs / "questions" / f"{cs_key}.parquet")
            report.n_questions[cs_key] = len(frame)

        for cs_key in cs_keys:
            qframe = question_sets[cs_key]

            responses = legacy.parse_responses(qframe, root)
            for model_key, group in responses.groupby(schema.MODEL, sort=True):
                out = outputs / "responses" / cs_key / f"{model_key}.parquet"
                write_table(group.reset_index(drop=True), out)
            report.n_responses[cs_key] = len(responses)

            for scale, frame in legacy.parse_judge_scores(qframe, root).items():
                out = outputs / "judge_scores" / cs_key / f"{legacy.LEGACY_JUDGE}_scale{scale}.parquet"
                write_table(frame, out)
                report.n_judge_scores[(cs_key, scale)] = len(frame)

            for encoder_key in legacy.discover_embedding_dirs(root):
                index_df, matrix = legacy.parse_embeddings(qframe, root, encoder_key)
                base = outputs / "embeddings" / cs_key / encoder_key
                write_table(index_df, base.with_suffix(".index.parquet"))
                write_matrix(matrix, base.with_suffix(".npy"))
                report.n_embeddings[(cs_key, encoder_key)] = len(index_df)

        from parallax_audit.runtime import record
        inputs = [p for section in ("raw_responses", "judged_responses", "embeddings")
                  for p in sorted((root / section).rglob("*")) if p.is_file()]
        inputs += sorted((registry.CONFIG_DIR / "questions").glob("*.csv"))
        record(outputs / "ingest.manifest.json", stage="archive-ingest", inputs=inputs,
               specification={"archive_judge_label": legacy.LEGACY_JUDGE,
                              "provenance": "reconstructed; historical deployment/judge identity unconfirmed"})
        report.defects = list(collector.records)
        return report
    finally:
        legacy_logger.removeHandler(collector)
