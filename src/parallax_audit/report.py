"""Recompute available archive analyses and export reproducible summary artifacts.

These are audit summary tables/figures, not a claim to regenerate every paper
figure or to fill missing historical experiments.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd

from parallax_audit import analysis, registry
from parallax_audit import data


def _latex(frame):
    def escape(value):
        return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
    lines = [r'\begin{tabular}{' + 'l' * len(frame.columns) + '}', r'\hline',
             ' & '.join(map(escape, frame.columns)) + r' \\', r'\hline']
    lines += [' & '.join(map(escape, row)) + r' \\' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines + [r'\hline', r'\end{tabular}']) + '\n'


def build(*, outputs: Path = Path('outputs'), tables=False, figures=False, B=20000, seed=0,
          archive_only=False, require_complete=False) -> Path:
    """Export registered results, optionally requiring the supported archive set."""
    if not tables and not figures:
        raise ValueError('report requires --tables and/or --figures')
    outputs = Path(outputs)
    embedding_rows, judge_rows, missing = [], [], []
    encoders = ('instructor', 'bge_large') if archive_only else registry.encoders()
    judges = ('gpt52',) if archive_only else registry.judges()
    for cs in sorted(registry.case_studies()):
        for encoder in encoders:
            base = outputs / 'embeddings' / cs / encoder
            if not base.with_suffix('.index.parquet').exists() or not base.with_suffix('.npy').exists():
                missing.append(f'embedding:{cs}:{encoder}')
                continue
            embedding_rows.append(analysis.embedding_analysis(cs, encoder, outputs=outputs))
        for judge in judges:
            for scale in (2, 4, 7, 10):
                path = outputs / 'judge_scores' / cs / f'{judge}_scale{scale}.parquet'
                if not path.exists():
                    missing.append(f'judge:{cs}:{judge}:scale{scale}')
                    continue
                judge_rows.append(analysis.judge_analysis(cs, judge, scale, outputs=outputs, B=B, seed=seed))
    if require_complete and missing:
        raise ValueError('missing required report inputs: ' + ', '.join(missing))
    if not embedding_rows and not judge_rows:
        raise ValueError('no report inputs found; run parallax_audit ingest first')
    destination = outputs / 'reports'
    destination.mkdir(parents=True, exist_ok=True)
    frames = {}
    for method, rows in [('embedding', embedding_rows), ('judge', judge_rows)]:
        if not rows:
            continue
        frame = pd.concat(rows, ignore_index=True)
        frames[method] = frame
        if tables:
            data.write_csv(frame, destination / f'{method}.csv')
            data.write_json(json.loads(frame.to_json(orient='records')), destination / f'{method}.json')
            data.write_text(_latex(frame), destination / f'{method}.tex')
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for method, frame in frames.items():
            labels = [f'{row.case_study}/{row.encoder}' if method == 'embedding'
                      else f'{row.case_study}/{row.judge}/{row.scale}' for row in frame.itertuples()]
            values = frame['diff' if method == 'embedding' else 'delta']
            fig, ax = plt.subplots(figsize=(max(7, len(labels) * .6), 4))
            ax.bar(range(len(values)), values)
            ax.set_xticks(range(len(labels)), labels, rotation=60, ha='right')
            ax.set_ylabel('Target minus baseline deviation')
            ax.axhline(0, color='black', linewidth=.8)
            fig.tight_layout()
            for extension in ('pdf', 'svg'):
                fig.savefig(destination / f'{method}.{extension}')
            plt.close(fig)
    inputs = {str(path.relative_to(outputs)): hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in ('questions', 'responses', 'embeddings', 'judge_scores')
              for path in sorted((outputs / folder).rglob('*')) if path.is_file() and path.suffix in ('.parquet', '.npy')}
    data.write_json({'scope': 'supported archive audit summaries' if archive_only else 'available original-eight-model audit summaries', 'missing': missing,
                      'resamples': B, 'seed': seed, 'input_sha256': inputs,
                      'rows': {name: len(frame) for name, frame in frames.items()}}, destination / 'manifest.json')
    return destination
