"""Shared response-collection loop, used by every provider.

:func:`collect_responses` implements retry, resume and concurrency once,
against the :class:`Provider` protocol, so a provider module only has to
implement ``generate``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

import pandas as pd

from parallax_audit import registry, schema
from parallax_audit.runtime import bind, read_records, delay as retry_delay, retryable

logger = logging.getLogger(__name__)

#: Columns a question set must carry in order to be queryable.
_QUESTION_COLUMNS = (schema.CASE_STUDY, schema.TOPIC, schema.QUESTION_ID, schema.QUESTION)


class Provider(Protocol):
    """Anything that can turn a prompt into a model response.

    Implementations (``bedrock.py``, ``deepseek.py``, ``metaai.py``,
    ``openrouter.py``) hide their SDK client behind this one method so that
    :func:`collect_responses` never needs to know which backend it is
    talking to.
    """

    def generate(self, prompt: str) -> str:
        """Return the model's response to ``prompt``. Raise on failure."""
        ...


def _call_with_retry(provider: Provider, prompt: str, *, max_retries: int) -> str | None:
    """Call ``provider.generate`` with exponential backoff plus jitter.

    Returns ``None`` (a "permanent failure") after ``max_retries`` attempts
    rather than raising, so one bad question never aborts a run of hundreds.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return provider.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - provider errors are opaque SDK exceptions
            last_exc = exc
            if attempt == max_retries - 1 or not retryable(exc):
                break
            delay = retry_delay(attempt)
            logger.warning(
                "generate() failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1,
                max_retries,
                last_exc,
                delay,
            )
            time.sleep(delay)
    logger.error("giving up after %d attempts: %s", max_retries, last_exc)
    return None


def _load_checkpoint(resume_path: Path, *, retry_failed: bool = True) -> dict[str, str]:
    """Read a JSONL checkpoint into ``{question_id: response}``.

    A missing or empty file just means "nothing done yet" -- not an error,
    since the first run of a query never has a checkpoint on disk.

    Permanently-failed questions are recorded with a null response so the
    checkpoint stays a faithful log of what was attempted. With
    ``retry_failed`` (the default) those nulls are *not* treated as done, so
    resuming retries them -- otherwise a transient provider outage would be
    frozen into the checkpoint and silently reproduced as missing data on
    every subsequent run.
    """
    done: dict[str, str] = {}
    if not resume_path.exists():
        return done
    for record in read_records(resume_path):
        response = record[schema.RESPONSE]
        question_id = record[schema.QUESTION_ID]
        if response is None and retry_failed:
            done.pop(question_id, None)
        else:
            done[question_id] = response
    return done


def collect_responses(
    questions_df: pd.DataFrame,
    provider: Provider,
    model_key: str,
    *,
    resume_path: str | Path | None = None,
    max_workers: int = 4,
    max_retries: int = 5,
    retry_failed: bool = True,
) -> pd.DataFrame:
    """Query ``provider`` for every row of ``questions_df``.

    ``questions_df`` must carry ``case_study``, ``topic``, ``question_id``
    and ``question`` columns (the shape of ``parallax_audit.data.load_questions()``
    output). The result conforms to :data:`schema.RESPONSES`.

    Resumable: completed ``question_id``s are appended to ``resume_path`` as
    JSONL as they finish, and skipped -- without calling ``provider`` -- on
    a subsequent run against the same path. Calls run concurrently via a
    thread pool since they are I/O-bound; a permanently-failing question
    (exhausts ``max_retries``) is recorded with a null response rather than
    raising, so the run always completes. Such failures are retried on the
    next resume unless ``retry_failed=False``.
    """
    missing = [c for c in _QUESTION_COLUMNS if c not in questions_df.columns]
    if missing:
        raise ValueError(f"questions_df is missing required columns: {missing}")

    if max_workers < 1 or max_retries < 1:
        raise ValueError("max_workers and max_retries must be positive")
    schema.validate(questions_df, _QUESTION_COLUMNS, name="questions")
    checkpoint_file = Path(resume_path) if resume_path is not None else None
    if checkpoint_file is not None:
        spec = registry.models().get(model_key)
        bind(checkpoint_file, questions_df, {"stage": "query", "model": model_key,
             "model_id": spec.model_id if spec else None, "provider": spec.provider if spec else None})
    done = (
        _load_checkpoint(checkpoint_file, retry_failed=retry_failed)
        if checkpoint_file is not None
        else {}
    )
    write_lock = threading.Lock()

    def _checkpoint_append(question_id: str, response: str | None) -> None:
        if checkpoint_file is None:
            return
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        record = {schema.QUESTION_ID: question_id, schema.RESPONSE: response}
        with write_lock:
            with checkpoint_file.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    pending = [row for row in questions_df.itertuples(index=False) if getattr(row, schema.QUESTION_ID) not in done]
    logger.info(
        "collecting responses for %s: %d question(s) total, %d already done, %d to run",
        model_key,
        len(questions_df),
        len(done),
        len(pending),
    )

    results: dict[str, str | None] = dict(done)
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_call_with_retry, provider, getattr(row, schema.QUESTION), max_retries=max_retries): getattr(
                    row, schema.QUESTION_ID
                )
                for row in pending
            }
            for future in as_completed(futures):
                question_id = futures[future]
                response = future.result()
                results[question_id] = response
                _checkpoint_append(question_id, response)

    records = []
    for row in questions_df.itertuples(index=False):
        question_id = getattr(row, schema.QUESTION_ID)
        records.append(
            {
                schema.CASE_STUDY: getattr(row, schema.CASE_STUDY),
                schema.TOPIC: getattr(row, schema.TOPIC),
                schema.QUESTION_ID: question_id,
                schema.QUESTION: getattr(row, schema.QUESTION),
                schema.MODEL: model_key,
                schema.RESPONSE: results.get(question_id),
            }
        )
    out = pd.DataFrame.from_records(records, columns=list(schema.RESPONSES))
    schema.validate(out, schema.RESPONSES, name="collect_responses output")
    return out
