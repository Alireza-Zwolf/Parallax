"""Opt-in regression against the preserved paper corpus."""
from pathlib import Path

import pandas as pd
import pytest

from parallax_audit import analysis
from parallax_audit.data import run_ingest
from parallax_audit.report import build

pytestmark = pytest.mark.archive


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    destination = tmp_path_factory.mktemp('archive')
    root = Path(__file__).resolve().parents[1]
    report = run_ingest(root=root / 'datasets', outputs=destination)
    assert report.n_questions == {'cs1': 100, 'cs2': 100, 'cs3': 50}
    assert sum(report.n_responses.values()) == 2000
    assert sum(report.n_judge_scores.values()) == 8000
    return destination


@pytest.mark.parametrize('cs,encoder,mu_t,mu_b,t,n', [
    ('cs1', 'instructor', .056105, .022674, 14.871519, 99),
    ('cs1', 'bge_large', .194978, .090026, 11.747469, 100),
    ('cs2', 'instructor', .029599, .027812, 1.404248, 100),
    ('cs2', 'bge_large', .114168, .112559, .348333, 100),
    ('cs3', 'instructor', .051994, .027211, 7.342783, 50),
    ('cs3', 'bge_large', .235751, .110115, 6.933052, 50),
])
def test_embedding_paper_results(artifacts, cs, encoder, mu_t, mu_b, t, n):
    row = analysis.embedding_analysis(cs, encoder, outputs=artifacts).iloc[0]
    assert row.mu_t == pytest.approx(mu_t, abs=1e-6)
    assert row.mu_b == pytest.approx(mu_b, abs=1e-6)
    assert row.t_stat == pytest.approx(t, abs=1e-5)
    assert row.n == n


def test_complete_report_exports_and_known_discrepancy(artifacts):
    path = build(outputs=artifacts, tables=True, figures=True)
    embedding = pd.read_csv(path / 'embedding.csv')
    judge = pd.read_csv(path / 'judge.csv')
    assert len(embedding) == 6
    assert len(judge) == 12
    row = judge[(judge.case_study == 'cs3') & (judge.scale == 10)].iloc[0]
    assert row.D_T == 3.5
    assert row.D_B == 0  # Recovered definition; historical paper reports .75.
    for name in ('embedding.tex', 'judge.json', 'embedding.pdf', 'judge.svg', 'manifest.json'):
        assert (path / name).stat().st_size > 0
