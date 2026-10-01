"""Artifact contracts and persistence failures that can change research results."""
import json

import numpy as np
import pandas as pd
import pytest

from parallax_audit import schema
from parallax_audit import data
from parallax_audit.runtime import bind, read_records


def scores():
    return pd.DataFrame([['cs1', 'topic', 'q1', 'model', 'judge', 4, 2, 'why']], columns=schema.JUDGE_SCORES)


def test_reject_duplicate_keys():
    with pytest.raises(ValueError, match='duplicate'):
        schema.validate(pd.concat([scores(), scores()]), schema.JUDGE_SCORES)


@pytest.mark.parametrize('score', [0, 5, 1.5, float('inf')])
def test_reject_invalid_scores(score):
    frame = scores().astype({schema.SCORE: float})
    frame[schema.SCORE] = score
    with pytest.raises(ValueError, match='scores'):
        schema.validate(frame, schema.JUDGE_SCORES)


def test_null_score_is_recorded_failure():
    frame = scores().astype({schema.SCORE: float})
    frame[schema.SCORE] = np.nan
    schema.validate(frame, schema.JUDGE_SCORES)


def test_reject_inconsistent_topics():
    frame = pd.concat([scores(), scores()], ignore_index=True)
    frame.loc[1, schema.MODEL] = 'other'
    frame.loc[1, schema.TOPIC] = 'different'
    with pytest.raises(ValueError, match='inconsistent'):
        schema.validate(frame, schema.JUDGE_SCORES)


def test_parquet_round_trip_has_real_format(tmp_path):
    path = tmp_path / 'table.parquet'
    data.write_table(scores(), path)
    assert path.read_bytes()[:4] == b'PAR1'
    pd.testing.assert_frame_equal(data.read_table(path), scores())


def test_failed_write_preserves_previous_artifact(tmp_path, monkeypatch):
    path = tmp_path / 'table.parquet'
    data.write_table(scores(), path)
    previous = path.read_bytes()
    def fail(*args, **kwargs):
        raise RuntimeError('disk failure')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', fail)
    with pytest.raises(RuntimeError):
        data.write_table(scores(), path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob('.table.parquet.*'))


@pytest.mark.parametrize('matrix', [np.zeros((1, 2)), np.array([[np.nan, 1]]), np.ones(2)])
def test_reject_invalid_embedding_matrix(matrix, tmp_path):
    with pytest.raises(ValueError):
        data.write_matrix(matrix, tmp_path / 'vectors.npy')


def test_checkpoint_rejects_changed_inputs_and_settings(tmp_path):
    path = tmp_path / 'checkpoint.jsonl'
    frame = scores()
    bind(path, frame, {'domain': 'old'})
    with pytest.raises(ValueError, match='different'):
        bind(path, frame, {'domain': 'new'})
    frame[schema.SCORE] = 3
    with pytest.raises(ValueError, match='different'):
        bind(path, frame, {'domain': 'old'})


def test_checkpoint_recovers_truncated_final_write(tmp_path):
    path = tmp_path / 'checkpoint.jsonl'
    path.write_text('{"done":1}\n{"bro')
    assert read_records(path) == [{'done': 1}]
    assert path.read_text() == '{"done":1}\n'


def test_checkpoint_rejects_corrupt_completed_record(tmp_path):
    path = tmp_path / 'checkpoint.jsonl'
    path.write_text('broken\n{"done":1}\n')
    with pytest.raises(ValueError, match='corrupt'):
        read_records(path)


def test_checkpoint_finishes_valid_record_without_newline(tmp_path):
    path = tmp_path / 'checkpoint.jsonl'
    path.write_text(json.dumps({'done': 1}))
    assert read_records(path) == [{'done': 1}]
    assert path.read_text().endswith('\n')
