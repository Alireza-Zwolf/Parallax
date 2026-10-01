"""Tests for parallax_audit.query and parallax_audit.cli.

Hard constraint: none of these tests may make a real LLM API call. Every
provider is exercised through a fake, or -- for the dispatch/dry-run tests
-- deliberately prevented from constructing a real client at all.
"""

from __future__ import annotations

import sys
import time

import pandas as pd
import pytest

from parallax_audit import cli, registry, schema
from parallax_audit.query.base import collect_responses
from parallax_audit.query.factory import get_provider

# --------------------------------------------------------------------------
# collect_responses
# --------------------------------------------------------------------------


class FakeProvider:
    """A Provider that counts calls and can be scripted to fail."""

    def __init__(self, *, fail_times: int = 0, always_fail: bool = False):
        self.call_count = 0
        self.fail_times = fail_times
        self.always_fail = always_fail

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        if self.always_fail:
            raise RuntimeError("permanent failure")
        if self.call_count <= self.fail_times:
            raise RuntimeError("transient failure")
        return f"answer to: {prompt}"


def _questions_df(n: int = 3, case_study: str = "cs1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            schema.CASE_STUDY: [case_study] * n,
            schema.TOPIC: ["topic-a"] * n,
            schema.QUESTION_ID: [f"{case_study}-q{i + 1:03d}" for i in range(n)],
            schema.QUESTION: [f"question {i}?" for i in range(n)],
        }
    )


def test_collect_responses_conforms_to_schema():
    provider = FakeProvider()
    questions_df = _questions_df(3)

    out = collect_responses(questions_df, provider, "fake_model", max_workers=2)

    assert tuple(out.columns) == schema.RESPONSES
    assert list(out[schema.MODEL].unique()) == ["fake_model"]
    assert out[schema.RESPONSE].notna().all()
    assert provider.call_count == 3


def test_collect_responses_resume_skips_completed(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    provider = FakeProvider()
    questions_df = _questions_df(4)

    first = collect_responses(questions_df, provider, "fake_model", resume_path=checkpoint)
    assert provider.call_count == 4
    assert first[schema.RESPONSE].notna().all()

    # Second run against the same checkpoint must not re-call the provider.
    second = collect_responses(questions_df, provider, "fake_model", resume_path=checkpoint)
    assert provider.call_count == 4  # unchanged
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )


def test_collect_responses_resume_continues_after_partial_run(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    questions_df = _questions_df(5)

    # Create a full-run identity, then simulate interruption after two writes.
    provider_a = FakeProvider()
    collect_responses(questions_df, provider_a, "fake_model", resume_path=checkpoint)
    lines = checkpoint.read_text().splitlines()
    checkpoint.write_text('\n'.join(lines[:2]) + '\n')
    # Resume with a fresh provider instance against the full question set.
    provider_b = FakeProvider()
    out = collect_responses(questions_df, provider_b, "fake_model", resume_path=checkpoint)
    assert provider_b.call_count == 3  # only the 3 not already checkpointed
    assert out[schema.RESPONSE].notna().all()
    assert len(out) == 5


def test_resume_retries_permanently_failed_questions(tmp_path, monkeypatch):
    """A provider outage must not be frozen into the checkpoint.

    Failures are logged as null so the checkpoint stays a faithful record of
    what was attempted, but a later resume has to retry them -- otherwise a
    transient outage silently becomes permanent missing data.
    """
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    checkpoint = tmp_path / "ckpt.jsonl"
    questions_df = _questions_df(3)

    failed = collect_responses(
        questions_df, FakeProvider(always_fail=True), "fake_model",
        resume_path=checkpoint, max_retries=2,
    )
    assert failed[schema.RESPONSE].isna().all()

    # Resuming with a healthy provider must re-attempt all three.
    healthy = FakeProvider()
    recovered = collect_responses(
        questions_df, healthy, "fake_model", resume_path=checkpoint
    )
    assert healthy.call_count == 3
    assert recovered[schema.RESPONSE].notna().all()


def test_resume_can_treat_failures_as_final(tmp_path, monkeypatch):
    """``retry_failed=False`` keeps nulls as done, for a deliberate give-up."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    checkpoint = tmp_path / "ckpt.jsonl"
    questions_df = _questions_df(2)

    collect_responses(
        questions_df, FakeProvider(always_fail=True), "fake_model",
        resume_path=checkpoint, max_retries=1,
    )
    healthy = FakeProvider()
    out = collect_responses(
        questions_df, healthy, "fake_model",
        resume_path=checkpoint, retry_failed=False,
    )
    assert healthy.call_count == 0
    assert out[schema.RESPONSE].isna().all()


def test_collect_responses_permanent_failure_yields_null_not_exception():
    provider = FakeProvider(always_fail=True)
    questions_df = _questions_df(2)

    out = collect_responses(questions_df, provider, "fake_model", max_retries=3)

    assert out[schema.RESPONSE].isna().all()
    assert tuple(out.columns) == schema.RESPONSES


def test_collect_responses_backoff_invoked_without_real_sleep(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))

    # Fails twice, succeeds on the third attempt -> two backoff sleeps.
    provider = FakeProvider(fail_times=2)
    questions_df = _questions_df(1)

    out = collect_responses(questions_df, provider, "fake_model", max_retries=5)

    assert out[schema.RESPONSE].iloc[0] is not None
    assert len(sleep_calls) == 2
    assert all(s >= 0 for s in sleep_calls)


def test_collect_responses_rejects_missing_columns():
    provider = FakeProvider()
    bad_df = pd.DataFrame({schema.QUESTION_ID: ["cs1-q001"]})
    with pytest.raises(ValueError, match="missing required columns"):
        collect_responses(bad_df, provider, "fake_model")


# --------------------------------------------------------------------------
# get_provider dispatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", sorted(registry.models()))
def test_get_provider_raises_actionable_error_without_sdk(model_key, monkeypatch):
    """Every model in models.yaml dispatches to *some* provider, and each
    provider fails loudly and actionably -- never a bare ImportError -- when
    its SDK is unavailable. Forcing every optional SDK to "not be installed"
    via sys.modules means this test never actually imports one, satisfying
    the no-network-call constraint regardless of what's on the test host.
    """
    for sdk_name in ("openai", "boto3", "meta_ai_api"):
        monkeypatch.setitem(sys.modules, sdk_name, None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        get_provider(model_key)

    message = str(exc_info.value)
    assert "pip install" in message


def test_get_provider_unknown_model_raises_key_error():
    with pytest.raises(KeyError):
        get_provider("not-a-real-model")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_main_help_exits_zero(capsys):
    assert cli.main(["--help"]) == 0


@pytest.mark.parametrize("subcommand", ["ingest", "query", "embed", "judge", "test", "report"])
def test_subcommand_help_exits_zero(subcommand, capsys):
    assert cli.main([subcommand, "--help"]) == 0


def test_query_dry_run_makes_zero_network_calls(monkeypatch, capsys):
    calls = []
    # parallax_audit.data / parallax_audit.query are only imported by cmd_query's non-dry-run
    # branch; if the dry-run path ever reached that import, this monkeypatch
    # would matter, and if it reached further to construct a client, boto3 /
    # openai / meta_ai_api aren't guaranteed present on this host and the
    # test would fail loudly rather than silently pass.
    monkeypatch.setattr(
        "parallax_audit.query.get_provider",
        lambda *a, **kw: calls.append((a, kw)) or (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    rc = cli.main(["query", "--model", "qwen3_max", "--case-study", "cs1", "--dry-run"])

    assert rc == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "qwen/qwen3-max" in out


def test_query_dry_run_all_eastern_and_all_case_studies(capsys):
    rc = cli.main(["query", "--model", "all-eastern", "--case-study", "all", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    for model_key in ("qwen3_max", "glm_52", "kimi_k2_6"):
        assert model_key in out
    for cs_key in ("cs1", "cs2", "cs3"):
        assert cs_key in out


def test_judge_dry_run_makes_zero_network_calls(capsys):
    # This command must not construct a real judge client.
    rc = cli.main(["judge", "--judge", "gpt52", "--case-study", "cs1", "--scale", "4", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out


def test_judge_dry_run_requires_scale_or_all_scales(capsys):
    rc = cli.main(["judge", "--judge", "gpt52", "--case-study", "cs1", "--dry-run"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "error:" in err


def test_unknown_model_key_clean_error(capsys):
    rc = cli.main(["query", "--model", "not-a-model", "--case-study", "cs1", "--dry-run"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "not-a-model" in err
    assert "Traceback" not in err


def test_unknown_case_study_key_clean_error(capsys):
    rc = cli.main(["query", "--model", "qwen3_max", "--case-study", "not-a-cs", "--dry-run"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err


def test_unknown_encoder_key_clean_error(capsys):
    rc = cli.main(["embed", "--encoder", "not-an-encoder", "--case-study", "cs1"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err
