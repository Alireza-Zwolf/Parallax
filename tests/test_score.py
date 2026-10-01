"""Tests for parallax_audit.score: rubrics, score parsing, the judge runner, and the
GPU embedding runner.

No test in this file makes a real LLM API call -- the judge tests use
in-process fakes that satisfy the ``JudgeClient`` protocol by duck typing.
The embedding test does make a real (local) GPU call to a downloaded
sentence-transformers checkpoint, which is the one piece of this module that
is expected to actually run.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from parallax_audit import schema
from parallax_audit.score.judge import score_responses
from parallax_audit.score.rubrics import parse_score, rubric

# --------------------------------------------------------------------------
# rubrics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [2, 4, 7, 10])
def test_rubric_scales_present_and_placeholders(scale):
    r = rubric(scale)
    assert r.scale == scale
    assert "{domain}" in r.template
    assert "{question}" in r.template
    assert "{response}" in r.template
    assert r.system_prompt  # non-empty, shared persona framing


def test_rubric_unknown_scale_raises():
    with pytest.raises(ValueError):
        rubric(3)


@pytest.mark.parametrize("scale", [2, 4, 7, 10])
def test_rubric_no_gemini_grounding_artifacts(scale):
    """The legacy prompts.py leaked literal `[cite: N]` markers into the live
    10-point prompt; none of the four rubrics here may contain any."""
    assert "[cite:" not in rubric(scale).template


def test_rubric_2point_title_is_not_4point():
    """Legacy bug: the binary rubric was titled '4-Point Evaluation Scale
    (Forced-Choice)'. It must say 2-point, not 4-point."""
    template = rubric(2).template
    assert "2-Point Evaluation Scale" in template
    assert "4-Point" not in template


def test_rubric_4point_title_is_4point():
    assert "4-Point Evaluation Scale" in rubric(4).template


def test_rubric_sha256_stable_and_distinguishes_scales():
    a = rubric(10)
    b = rubric(10)
    assert a.sha256 == b.sha256  # same content -> same hash, every call
    assert a.sha256 == a.sha256  # stable across repeated access
    hashes = {rubric(s).sha256 for s in (2, 4, 7, 10)}
    assert len(hashes) == 4  # four distinct templates


# --------------------------------------------------------------------------
# parse_score
# --------------------------------------------------------------------------

_PARSE_CASES = [
    pytest.param(
        "Evaluation Score: 7\nExplanation: Clear and direct critique.",
        10,
        (7, "Clear and direct critique."),
        id="well_formed",
    ),
    pytest.param(
        "evaluation score: 3\nexplanation: lowercase headers still parse.",
        7,
        (3, "lowercase headers still parse."),
        id="lowercase_headers",
    ),
    pytest.param(
        "Evaluation Score: 15\nExplanation: score exceeds the scale.",
        10,
        (None, "score exceeds the scale."),
        id="out_of_range_high",
    ),
    pytest.param(
        "Evaluation Score: 0\nExplanation: score is below the scale.",
        4,
        (None, "score is below the scale."),
        id="out_of_range_low",
    ),
    pytest.param(
        "This response is fairly neutral overall.",
        10,
        (None, ""),
        id="missing_score_line",
    ),
    pytest.param(
        "Evaluation Score: 2\nExplanation: First line.\nSecond line.\n\nThird paragraph.",
        4,
        (2, "First line.\nSecond line.\n\nThird paragraph."),
        id="multiline_explanation",
    ),
    pytest.param(
        "Evaluation Score: 5 (this is a borderline case)\nExplanation: trailing prose after the score.",
        7,
        (5, "trailing prose after the score."),
        id="score_with_trailing_prose",
    ),
    pytest.param(
        "Evaluation Score: 1\nExplanation: minimal but valid.",
        2,
        (1, "minimal but valid."),
        id="binary_scale_boundary",
    ),
]


@pytest.mark.parametrize("text,scale,expected", _PARSE_CASES)
def test_parse_score_table(text, scale, expected):
    assert parse_score(text, scale) == expected


# --------------------------------------------------------------------------
# judge (FakeJudge -- no network calls)
# --------------------------------------------------------------------------


class FakeJudge:
    """Returns a canned, well-formed completion and counts calls."""

    def __init__(self, score: int = 3) -> None:
        self.score = score
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return f"Evaluation Score: {self.score}\nExplanation: canned response."


class CrashingFakeJudge:
    """Succeeds `succeed_for` times, then raises on every call after that --
    simulating a transport failure that persists across retries."""

    def __init__(self, succeed_for: int, score: int = 1) -> None:
        self.succeed_for = succeed_for
        self.score = score
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls > self.succeed_for:
            raise RuntimeError("simulated transport failure")
        return f"Evaluation Score: {self.score}\nExplanation: ok before the crash."


class GarbageJudge:
    """Never produces a parseable score."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return "I refuse to follow the requested format."


def _make_rows(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            schema.CASE_STUDY: ["cs1"] * n,
            schema.TOPIC: ["Internet Censorship"] * n,
            schema.QUESTION_ID: [f"cs1-q{i:03d}" for i in range(1, n + 1)],
            schema.QUESTION: [f"Question {i}?" for i in range(1, n + 1)],
            schema.MODEL: ["deepseek_r1"] * n,
            schema.RESPONSE: [f"Response text {i}." for i in range(1, n + 1)],
        }
    )[list(schema.RESPONSES)]


def test_score_responses_conforms_to_schema():
    rows = _make_rows(3)
    judge = FakeJudge(score=4)
    frame = score_responses(rows, judge, scale=7, domain="test domain", judge_key="gpt52")
    assert tuple(frame.columns) == schema.JUDGE_SCORES
    schema.validate(frame, schema.JUDGE_SCORES)
    assert len(frame) == 3
    assert judge.calls == 3
    assert set(frame[schema.SCORE]) == {4}
    assert set(frame[schema.JUDGE]) == {"gpt52"}
    assert set(frame[schema.SCALE]) == {7}


def test_score_responses_domain_and_question_reach_the_judge():
    """The rendered prompt must carry the caller's domain, not a hardcoded
    one -- that was the legacy bug this module fixes."""
    seen_prompts: list[str] = []

    class RecordingJudge:
        def complete(self, system: str, user: str) -> str:
            seen_prompts.append(user)
            return "Evaluation Score: 1\nExplanation: ok."

    rows = _make_rows(1)
    score_responses(rows, RecordingJudge(), scale=2, domain="a very specific domain", judge_key="gpt52")
    assert len(seen_prompts) == 1
    assert "a very specific domain" in seen_prompts[0]
    assert "Question 1?" in seen_prompts[0]
    assert "Response text 1." in seen_prompts[0]


def test_score_responses_resume_skips_completed_rows_after_a_crash(tmp_path):
    rows = _make_rows(3)
    checkpoint = tmp_path / "checkpoint.jsonl"

    crasher = CrashingFakeJudge(succeed_for=1)  # row 1 succeeds, row 2 raises forever
    with pytest.raises(RuntimeError, match="simulated transport failure"):
        score_responses(
            rows,
            crasher,
            scale=4,
            domain="d",
            judge_key="gpt52",
            resume_path=checkpoint,
            max_workers=1,  # deterministic FIFO processing order
            max_retries=1,  # no backoff sleep needed to prove the point
        )

    # Exactly the one row that finished before the crash was checkpointed.
    persisted = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert len(persisted) == 1
    assert persisted[0][schema.QUESTION_ID] == "cs1-q001"

    good_judge = FakeJudge(score=2)
    frame = score_responses(
        rows,
        good_judge,
        scale=4,
        domain="d",
        judge_key="gpt52",
        resume_path=checkpoint,
        max_workers=1,
        max_retries=1,
    )

    # Only the two rows that were NOT already checkpointed triggered a call.
    assert good_judge.calls == 2
    assert len(frame) == 3
    schema.validate(frame, schema.JUDGE_SCORES)
    assert set(frame[schema.QUESTION_ID]) == {"cs1-q001", "cs1-q002", "cs1-q003"}
    # Row 1's score came from the checkpoint (the crasher's score=1), not
    # from the good judge that only ran for rows 2 and 3.
    row1 = frame[frame[schema.QUESTION_ID] == "cs1-q001"].iloc[0]
    assert row1[schema.SCORE] == 1


def test_score_responses_unparseable_response_yields_null_not_abort():
    rows = _make_rows(2)
    garbage = GarbageJudge()
    frame = score_responses(
        rows,
        garbage,
        scale=10,
        domain="d",
        judge_key="gemini31_flash_lite",
        max_retries=1,  # no backoff sleep needed to prove the point
    )
    schema.validate(frame, schema.JUDGE_SCORES)
    assert len(frame) == 2
    assert frame[schema.SCORE].isna().all()
    assert garbage.calls == 2  # one attempt per row, no exception raised


def test_score_responses_rejects_wrong_schema():
    bad_rows = pd.DataFrame({"not": ["a"], "schema": ["match"]})
    with pytest.raises(ValueError):
        score_responses(bad_rows, FakeJudge(), scale=4, domain="d", judge_key="gpt52")


def test_score_responses_rejects_unsupported_scale():
    rows = _make_rows(1)
    with pytest.raises(ValueError):
        score_responses(rows, FakeJudge(), scale=3, domain="d", judge_key="gpt52")


# --------------------------------------------------------------------------
# embed (real GPU call, no mocking)
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_embed_responses_minilm_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    from parallax_audit.score.embed import embed_responses

    texts = [
        "The government announced new regulations today.",
        "Scientists discovered a new species of frog.",
        "The stock market fell sharply after the announcement.",
        "A local bakery won an award for its sourdough bread.",
        "Protesters gathered outside the courthouse this morning.",
        "The new smartphone features a faster processor.",
        "Researchers published a study on climate change.",
        "The government announced new regulations today.",  # duplicate of index 0
    ]

    vectors = embed_responses(texts, "minilm_l6", batch_size=64, device="cuda")

    assert vectors.shape == (8, 384)
    assert vectors.dtype == np.float32

    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Identical input text -> identical vectors.
    assert np.allclose(vectors[0], vectors[-1], atol=1e-5)
    # Distinct texts should not collapse to the same vector.
    assert not np.allclose(vectors[0], vectors[1], atol=1e-3)


def test_embed_responses_rejects_unknown_instruction_style_dim_mismatch(monkeypatch):
    """A defensive check: if an encoder's declared dim disagrees with what
    the model actually produces, embed_responses must raise, not silently
    hand back the wrong shape."""
    from parallax_audit import registry
    from parallax_audit.score import embed as embed_mod

    real_encoder = registry.encoder("minilm_l6")
    bad_encoder = registry.Encoder(
        key=real_encoder.key,
        display=real_encoder.display,
        model_id=real_encoder.model_id,
        dim=999,  # wrong on purpose
        instruction_style=real_encoder.instruction_style,
        aliases=real_encoder.aliases,
    )
    monkeypatch.setattr(registry, "encoder", lambda key: bad_encoder)
    monkeypatch.setattr(embed_mod, "registry", registry)
    class FakeEncoder:
        def encode(self, inputs, **kwargs):
            return np.ones((len(inputs), 384), dtype=np.float32)
    monkeypatch.setattr(embed_mod, "_load_model", lambda *args: FakeEncoder())

    with pytest.raises(ValueError, match="expected"):
        embed_mod.embed_responses(["hello"], "minilm_l6", device="cpu")
