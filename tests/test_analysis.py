"""Model-pool selection must precede shared-question intersection."""
import numpy as np
import pandas as pd
import pytest

from parallax_audit import analysis, registry, schema, data


def embedding_fixture():
    selected = ['deepseek_r1', 'claude_37_sonnet', 'mistral_large']
    rows = []
    for model in [*selected, 'qwen3_max']:
        for q in (['q1'] if model == 'qwen3_max' else ['q1', 'q2', 'q3']):
            rows.append(['cs1', 'topic', q, model, 'instructor', len(rows)])
    return selected, pd.DataFrame(rows, columns=schema.EMBEDDING_INDEX), np.ones((len(rows), 2))


def test_unrelated_missing_questions_do_not_shrink_fixed_pool(monkeypatch):
    selected, index, vectors = embedding_fixture()
    monkeypatch.setattr(data, 'load_embeddings', lambda *args, **kwargs: (index, vectors))
    matrices, questions, _ = analysis.embedding_matrices('instructor', 'cs1', models=selected)
    assert set(matrices) == set(selected)
    assert questions == ['q1', 'q2', 'q3']


def test_missing_required_model_is_error(monkeypatch):
    selected, index, vectors = embedding_fixture()
    monkeypatch.setattr(data, 'load_embeddings', lambda *args, **kwargs: (index, vectors))
    with pytest.raises(ValueError, match='missing required'):
        analysis.embedding_matrices('instructor', 'cs1', models=selected + ['glm_52'])


def test_target_swap_uses_selected_pool(monkeypatch):
    selected, index, vectors = embedding_fixture()
    monkeypatch.setattr(data, 'load_embeddings', lambda *args, **kwargs: (index, vectors))
    monkeypatch.setattr(registry, 'pool', lambda key: registry.Pool('small', 'Small', '', tuple(selected)))
    seen = []
    def permute(dist, universe, **kwargs):
        seen.extend(universe)
        return pd.DataFrame({'target': universe, 'p_value': [.5]*len(universe)})
    monkeypatch.setattr(analysis.stats, 'permute_embedding_targets', permute)
    analysis.embedding_analysis('cs1', 'instructor', pool='small', permute=True)
    assert set(seen) == set(selected)


def test_default_pool_is_frozen_to_original_eight():
    assert set(analysis._members('deepseek_r1', None)) == set(registry.original_eight())
