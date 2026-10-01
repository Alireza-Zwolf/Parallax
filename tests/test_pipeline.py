"""Regression checks for public command wiring and keyed agreement."""
import pandas as pd
import pytest

from parallax_audit import analysis, cli, pipeline, data, schema
from parallax_audit.score import judge


def response_rows():
    return pd.DataFrame([['cs1', 'topic', 'cs1-q001', 'Question?', 'deepseek_r1', 'Answer.']],
                        columns=schema.RESPONSES)


def test_judge_command_uses_configured_client_and_domain(tmp_path, monkeypatch):
    class FakeJudge:
        def complete(self, system, user):
            assert 'political bias and censorship' in user
            assert 'Question?' in user
            return 'Evaluation Score: 2\nExplanation: Example.'

    monkeypatch.setattr(data, 'load_responses', lambda cs, **kwargs: response_rows())
    monkeypatch.setattr(judge, 'get_judge_client', lambda key: FakeJudge())

    out = tmp_path / 'scores.parquet'
    assert cli.main(['judge', '--judge', 'gpt52', '--case-study', 'cs1',
                     '--scale', '2', '--resume', '--out', str(out)]) == 0
    assert data.read_table(out)[schema.SCORE].tolist() == [2]
    assert out.with_suffix('.ckpt.jsonl').exists()


def test_report_without_inputs_returns_failure(tmp_path, capsys):
    assert cli.main(['report', '--tables', '--outputs', str(tmp_path)]) == 1
    assert 'no report inputs' in capsys.readouterr().err


def test_archive_run_requires_local_source(tmp_path, capsys):
    assert cli.main(['archive-run', '--source', str(tmp_path / 'absent'),
                     '--outputs', str(tmp_path / 'out')]) == 1
    assert 'archive source does not exist' in capsys.readouterr().err


def test_archive_run_passes_one_output_root_through_stages(tmp_path, monkeypatch):
    from parallax_audit import data as ingest
    from parallax_audit import report

    source = tmp_path / 'archive'
    source.mkdir()
    output = tmp_path / 'artifacts'
    seen = {}

    def fake_ingest(*, root, outputs):
        seen['ingest'] = (root, outputs)
        keys = ('cs1', 'cs2', 'cs3')
        counts = {'cs1': 100, 'cs2': 100, 'cs3': 50}
        return ingest.IngestReport(
            n_questions=counts,
            n_responses={key: 8 * counts[key] for key in keys},
            n_judge_scores={(key, scale): 8 * counts[key] for key in keys for scale in (2, 4, 7, 10)},
            n_embeddings={(key, encoder): 8 * counts[key] - int(key == 'cs1' and encoder == 'instructor')
                          for key in keys for encoder in ('instructor', 'bge_large')},
        )

    def fake_build(**kwargs):
        seen['build'] = kwargs
        return output / 'reports'

    def fake_record(path, **kwargs):
        seen['record'] = (path, kwargs)

    monkeypatch.setattr(ingest, 'run_ingest', fake_ingest)
    monkeypatch.setattr(report, 'build', fake_build)
    monkeypatch.setattr(pipeline, 'record', fake_record)
    assert pipeline.archive_run(source=source, outputs=output, resamples=50, seed=3, figures=False) == output / 'reports'
    assert seen['ingest'] == (source, output)
    assert seen['build'] == dict(outputs=output, tables=True, figures=False, B=50,
                                 seed=3, archive_only=True, require_complete=True)
    assert seen['record'][0] == output / 'reports' / 'archive-run.manifest.json'


def test_judge_checkpoint_creates_parent_directory(tmp_path):
    class FakeJudge:
        def complete(self, system, user):
            return 'Evaluation Score: 1\nExplanation: Example.'

    checkpoint = tmp_path / 'nested' / 'scores.jsonl'
    scores = judge.score_responses(response_rows(), FakeJudge(), 2, 'domain', 'gpt52',
                                    resume_path=checkpoint)
    assert checkpoint.exists()
    assert scores[schema.SCORE].tolist() == [1]


def test_agreement_matches_item_ids_with_different_missing_questions(monkeypatch):
    def load(case_study, judge, scale, **kwargs):
        values = ([('q1', 1), ('q2', 4), ('q3', 2)] if judge == 'a'
                  else [('q4', 3), ('q3', 2), ('q1', 1)])
        return pd.DataFrame([['cs1', 'topic', q, 'model', judge, 4, score, '']
                             for q, score in values], columns=schema.JUDGE_SCORES)
    monkeypatch.setattr(data, 'load_judge_scores', load)
    result = analysis.agreement_analysis('cs1', 4, 'a', 'b')
    assert result.iloc[0]['value'] == pytest.approx(1)
    assert result.iloc[0]['n'] == 2


def test_agreement_rejects_duplicate_item_ids(monkeypatch):
    frame = pd.DataFrame([['cs1', 'topic', 'q1', 'model', 'a', 4, 1, '']] * 2,
                         columns=schema.JUDGE_SCORES)
    monkeypatch.setattr(data, 'load_judge_scores', lambda **kw: frame)
    with pytest.raises(ValueError, match='duplicate'):
        analysis.agreement_analysis('cs1', 4, 'a', 'b')


def test_batch_out_cannot_overwrite_artifacts(tmp_path, capsys):
    assert cli.main(['query', '--model', 'all-eastern', '--case-study', 'all',
                     '--dry-run', '--out', str(tmp_path / 'one.parquet')]) == 1
    assert 'exactly one' in capsys.readouterr().err


def test_missing_inputs_are_clean_cli_errors(tmp_path, capsys):
    assert cli.main(['test', '--method', 'embedding', '--case-study', 'cs1',
                     '--outputs', str(tmp_path)]) == 1
    assert 'error:' in capsys.readouterr().err


def test_judge_retries_null_checkpoint_scores(tmp_path):
    class FakeJudge:
        def __init__(self, text):
            self.text, self.calls = text, 0
        def complete(self, system, user):
            self.calls += 1
            return self.text
    path = tmp_path / 'checkpoint.jsonl'
    failed = FakeJudge('garbage')
    judge.score_responses(response_rows(), failed, 2, 'domain', 'gpt52', resume_path=path, max_retries=1)
    recovered = FakeJudge('Evaluation Score: 2\nExplanation: ok')
    frame = judge.score_responses(response_rows(), recovered, 2, 'domain', 'gpt52', resume_path=path)
    assert recovered.calls == 1
    assert frame[schema.SCORE].tolist() == [2]
