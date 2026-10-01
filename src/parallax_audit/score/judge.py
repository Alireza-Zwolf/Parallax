"""Provider-agnostic LLM-as-a-Judge runner.

A judging run is thousands of paid API calls, so every scored row is appended
to a JSONL checkpoint as it completes and a restart skips whatever
``(question_id, model, scale)`` keys are already there. Calls are I/O-bound
and run on a thread pool. A transport failure is retried with exponential
backoff and then re-raised; a response that never parses to a valid score is
retried and then recorded as ``score=None`` so one bad row never aborts a run.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

import pandas as pd

from parallax_audit import registry, schema
from parallax_audit.runtime import bind, read_records, delay as retry_delay, retryable
from parallax_audit.score.rubrics import parse_score, rubric as get_rubric

logger = logging.getLogger(__name__)


class JudgeClient(Protocol):
    """One judge round-trip. Providers own auth and request construction;
    :func:`score_responses` owns retry, backoff and resume, so every
    provider gets identical semantics for free."""

    def complete(self, system: str, user: str) -> str:
        """Return the raw completion text for one (system, user) prompt."""
        ...


class OpenAIJudge:
    """:class:`JudgeClient` backed by the OpenAI chat.completions API."""

    def __init__(self, judge_key: str = "gpt52") -> None:
        from openai import OpenAI  # lazy: don't require `openai` just to import this module

        self._judge = registry.judge(judge_key)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        # No `temperature` is sent: reasoning-family models may reject it.
        self._client = OpenAI(api_key=key, timeout=60.0, max_retries=0)

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._judge.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class GeminiJudge:
    """:class:`JudgeClient` backed by the Google Generative AI SDK."""

    def __init__(self, judge_key: str = "gemini31_flash_lite") -> None:
        import google.generativeai as genai  # lazy: don't require this SDK to import the module

        self._judge = registry.judge(judge_key)
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=key)
        self._genai = genai

    def complete(self, system: str, user: str) -> str:
        # A fresh GenerativeModel per call is cheap (no network I/O at
        # construction) and lets `system` vary per call as the Protocol
        # requires, rather than baking one system prompt in at __init__.
        model = self._genai.GenerativeModel(self._judge.model_id, system_instruction=system)
        response = model.generate_content(user)
        return response.text


def get_judge_client(judge_key: str) -> JudgeClient:
    """Construct the configured provider adapter for a judge."""
    spec = registry.judge(judge_key)
    if spec.provider == "openai":
        return OpenAIJudge(judge_key)
    if spec.provider == "gemini":
        return GeminiJudge(judge_key)
    raise ValueError(f"unsupported judge provider: {spec.provider!r}")


# --------------------------------------------------------------------------
# Retry / backoff
# --------------------------------------------------------------------------


def _score_one(
    rubric_text: str,
    system_prompt: str,
    client: JudgeClient,
    scale: int,
    max_retries: int,
) -> tuple[int | None, str]:
    """Run the retry policy for a single (question, response) pair.

    A transport exception is retried with backoff and re-raised after
    ``max_retries`` attempts -- that propagates out of the worker thread and
    aborts the whole run, which is correct: it signals an operational
    problem (bad key, outage, quota) rather than a per-row issue.

    A response that comes back but fails :func:`~parallax_audit.score.rubrics.parse_score`
    is also retried up to ``max_retries`` times (hoping for a better-formed
    completion), but exhausting those retries is NOT fatal: the row is
    returned with ``score=None`` so the run continues.
    """
    explanation = ""
    for attempt in range(max_retries):
        try:
            raw = client.complete(system_prompt, rubric_text)
        except Exception as exc:
            if attempt == max_retries - 1 or not retryable(exc):
                raise
            logger.warning("judge call failed (attempt %d/%d); retrying", attempt + 1, max_retries)
            time.sleep(retry_delay(attempt))
            continue

        score, explanation = parse_score(raw, scale)
        if score is not None:
            return score, explanation
        if attempt < max_retries - 1:
            logger.warning(
                "judge response did not parse to a valid 1..%d score (attempt %d/%d); retrying",
                scale,
                attempt + 1,
                max_retries,
            )
            time.sleep(retry_delay(attempt))

    logger.error("giving up after %d attempts; recording a null score", max_retries)
    return None, explanation


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def score_responses(
    rows: pd.DataFrame,
    client: JudgeClient,
    scale: int,
    domain: str,
    judge_key: str,
    *,
    resume_path: str | Path | None = None,
    max_workers: int = 8,
    max_retries: int = 5,
    retry_failed: bool = True,
) -> pd.DataFrame:
    """Score every row of ``rows`` (schema.RESPONSES) with ``client``.

    ``judge_key`` labels the output rows' ``schema.JUDGE`` column; it is
    metadata the caller supplies (e.g. ``"gpt52"``) rather than something
    inferred from ``client``, so any object satisfying :class:`JudgeClient`
    -- including a test fake -- can be used without also having to impersonate
    a registry entry.

    Resumability: if ``resume_path`` is given, results are appended to it as
    JSONL as each row finishes, and rows whose ``(question_id, model, scale)``
    key is already present are skipped. Pass ``resume_path=None`` to disable
    checkpointing (e.g. in tests that don't need it); in that case nothing is
    written to disk and a crash loses all progress from that call.

    Concurrency is via ``ThreadPoolExecutor(max_workers=max_workers)``: these
    are I/O-bound HTTP calls, so threads (not processes) are the right
    primitive and avoid re-pickling the client/rubric per task.
    """
    schema.validate(rows, schema.RESPONSES, name="rows")
    if scale not in schema.VALID_SCALES:
        raise ValueError(f"unsupported scale {scale!r}; expected one of {schema.VALID_SCALES}")

    if max_workers < 1 or max_retries < 1:
        raise ValueError("max_workers and max_retries must be positive")
    rb = get_rubric(scale)
    checkpoint_path = Path(resume_path) if resume_path is not None else None
    if checkpoint_path is not None:
        spec = registry.judges().get(judge_key)
        bind(checkpoint_path, rows, {"stage": "judge", "judge": judge_key,
             "model_id": spec.model_id if spec else None, "scale": scale, "domain": domain,
             "rubric": rb.sha256, "system_prompt": rb.system_prompt})
    completed: list[dict] = read_records(checkpoint_path) if checkpoint_path is not None else []
    latest = {(r[schema.QUESTION_ID], r[schema.MODEL], r[schema.SCALE]): r for r in completed}
    completed = [r for r in latest.values() if not retry_failed or r[schema.SCORE] is not None]
    done_keys = {(r[schema.QUESTION_ID], r[schema.MODEL], r[schema.SCALE]) for r in completed}

    row_list = list(rows.itertuples(index=False))
    pending = [
        row
        for row in row_list
        if (getattr(row, schema.QUESTION_ID), getattr(row, schema.MODEL), scale) not in done_keys
    ]
    logger.info(
        "scoring %d/%d rows at scale=%d (%d already checkpointed)",
        len(pending),
        len(row_list),
        scale,
        len(row_list) - len(pending),
    )

    write_lock = threading.Lock()

    def _persist(record: dict) -> None:
        completed.append(record)
        if checkpoint_path is None:
            return
        with write_lock, checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()

    def _work(row) -> dict:
        question = getattr(row, schema.QUESTION)
        response_text = getattr(row, schema.RESPONSE)
        if pd.isna(response_text):
            score, explanation = None, "No response available; judge was not called."
        else:
            user_prompt = rb.template.format(domain=domain, question=question, response=response_text)
            score, explanation = _score_one(user_prompt, rb.system_prompt, client, scale, max_retries)
        return {
            schema.CASE_STUDY: getattr(row, schema.CASE_STUDY),
            schema.TOPIC: getattr(row, schema.TOPIC),
            schema.QUESTION_ID: getattr(row, schema.QUESTION_ID),
            schema.MODEL: getattr(row, schema.MODEL),
            schema.JUDGE: judge_key,
            schema.SCALE: scale,
            schema.SCORE: score,
            schema.EXPLANATION: explanation,
        }

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_work, row) for row in pending]
            for future in as_completed(futures):
                record = future.result()  # re-raises a non-transient client failure here
                _persist(record)

    wanted = {(getattr(row, schema.QUESTION_ID), getattr(row, schema.MODEL)) for row in row_list}
    kept = [
        r
        for r in completed
        if r[schema.SCALE] == scale and (r[schema.QUESTION_ID], r[schema.MODEL]) in wanted
    ]
    frame = pd.DataFrame.from_records(kept, columns=list(schema.JUDGE_SCORES))
    frame = frame.sort_values([schema.CASE_STUDY, schema.QUESTION_ID, schema.MODEL]).reset_index(drop=True)
    schema.validate(frame, schema.JUDGE_SCORES, name="scored")
    return frame
