"""Application services: validated inputs, one stage, persisted artifacts.

The CLI supplies explicit paths. Scientific calculations live in stats/;
provider and encoder imports are deferred until the corresponding stage runs.
"""
from pathlib import Path

from parallax_audit import registry, schema
from parallax_audit import data
from parallax_audit.runtime import record


def archive_run(*, source: Path = Path('datasets'), outputs: Path = Path('outputs'),
                resamples: int = 20000, seed: int = 0, figures: bool = True) -> Path:
    """Recompute the supported archive summaries from local files only.

    This path deliberately uses stored embeddings and judge scores. It never
    constructs an encoder or provider client.
    """
    from parallax_audit.report import build
    from parallax_audit.data import run_ingest

    source, outputs = Path(source), Path(outputs)
    if not source.is_dir():
        raise FileNotFoundError(f'archive source does not exist: {source}')
    if resamples <= 0:
        raise ValueError('resamples must be positive')
    ingest = run_ingest(root=source, outputs=outputs)
    expected = set(registry.case_studies())
    for name, observed in (
        ('questions', set(ingest.n_questions)),
        ('responses', set(ingest.n_responses)),
        ('judge scores', {cs for cs, _ in ingest.n_judge_scores}),
        ('embeddings', {cs for cs, _ in ingest.n_embeddings}),
    ):
        if observed != expected:
            raise ValueError(f'incomplete archive {name}: expected {sorted(expected)}, found {sorted(observed)}')
    for cs in sorted(expected):
        questions = registry.case_study(cs).n_questions
        if ingest.n_questions[cs] != questions:
            raise ValueError(f'{cs}: expected {questions} questions, found {ingest.n_questions[cs]}')
        rows = questions * len(registry.original_eight())
        if ingest.n_responses[cs] != rows:
            raise ValueError(f'{cs}: expected {rows} responses, found {ingest.n_responses[cs]}')
        for scale in schema.VALID_SCALES:
            observed = ingest.n_judge_scores.get((cs, scale))
            if observed != rows:
                raise ValueError(f'{cs} scale {scale}: expected {rows} judge scores, found {observed}')
        for encoder in ('instructor', 'bge_large'):
            observed = ingest.n_embeddings.get((cs, encoder))
            # One historical DeepSeek-R1/CS1 Instructor vector is absent.
            expected_rows = rows - int(cs == 'cs1' and encoder == 'instructor')
            if observed != expected_rows:
                raise ValueError(f'{cs}/{encoder}: expected {expected_rows} embeddings, found {observed}')
    destination = build(outputs=outputs, tables=True, figures=figures, B=resamples,
                        seed=seed, archive_only=True, require_complete=True)
    record(destination / 'archive-run.manifest.json', stage='archive-run',
           inputs=[outputs / 'ingest.manifest.json', destination / 'manifest.json'],
           specification={
               'source': str(source.resolve()), 'outputs': str(outputs.resolve()),
               'encoders': ['instructor', 'bge_large'], 'judge': 'gpt52',
               'scales': [2, 4, 7, 10], 'resamples': resamples, 'seed': seed,
               'questions': ingest.n_questions, 'responses': ingest.n_responses,
               'judge_scores': {f'{cs}:scale{scale}': n for (cs, scale), n in ingest.n_judge_scores.items()},
               'embeddings': {f'{cs}:{encoder}': n for (cs, encoder), n in ingest.n_embeddings.items()},
               'ingest_defects': ingest.defects,
           })
    return destination


def collect(model: str, case_study: str, *, outputs: Path, out: Path | None = None,
            resume: bool = False, max_workers: int = 4) -> Path:
    from parallax_audit.query import collect_responses, get_provider
    questions = data.load_questions(case_study, outputs=outputs)
    path = out or outputs / 'responses' / case_study / f'{model}.parquet'
    frame = collect_responses(questions, get_provider(model), model,
                              resume_path=path.with_suffix('.ckpt.jsonl') if resume else None,
                              max_workers=max_workers)
    data.write_table(frame, path)
    record(path.with_suffix(".manifest.json"), stage="query",
           inputs=[outputs / "questions" / f"{case_study}.parquet"], specification=registry.model(model))
    if frame[schema.RESPONSE].isna().any():
        raise RuntimeError(f'{path}: incomplete collection; use --resume to retry failed rows')
    return path


def embed(encoder: str, case_study: str, *, outputs: Path, out: Path | None = None,
          device: str | None = None, batch_size: int = 64) -> Path:
    from parallax_audit.score.embed import embed_responses
    frame = data.load_responses(case_study, outputs=outputs)
    if frame[schema.RESPONSE].isna().any():
        raise ValueError('responses contain failed rows; repair collection before embedding')
    vectors = embed_responses(frame[schema.RESPONSE].tolist(), encoder,
                              instruction=registry.case_study(case_study).embed_instruction,
                              device=device, batch_size=batch_size)
    base = out or outputs / 'embeddings' / case_study / encoder
    index = frame[[schema.CASE_STUDY, schema.TOPIC, schema.QUESTION_ID, schema.MODEL]].copy()
    index[schema.ENCODER] = encoder
    index[schema.ROW] = range(len(index))
    schema.validate(index, schema.EMBEDDING_INDEX, name='embedding index')
    data.write_matrix(vectors, base.with_suffix('.npy'))
    data.write_table(index, base.with_suffix('.index.parquet'))
    record(base.with_suffix(".manifest.json"), stage="embedding",
           inputs=sorted((outputs / "responses" / case_study).glob("*.parquet")),
           specification={"encoder": encoder, "model_id": registry.encoder(encoder).model_id,
                          "instruction": registry.case_study(case_study).embed_instruction,
                          "batch_size": batch_size, "device": device})
    return base


def judge(judge_key: str, case_study: str, scale: int, *, outputs: Path,
          out: Path | None = None, resume: bool = False, max_workers: int = 8,
          allow_unconfirmed_domain: bool = False) -> Path:
    from parallax_audit.score.judge import get_judge_client, score_responses
    spec = registry.case_study(case_study)
    if not spec.judge_domain_confirmed and not allow_unconfirmed_domain:
        raise ValueError(f'{case_study}: historical judge domain is unconfirmed; '
                         'confirm configuration or use --allow-unconfirmed-domain for a new experiment')
    rows = data.load_responses(case_study, outputs=outputs)
    path = out or outputs / 'judge_scores' / case_study / f'{judge_key}_scale{scale}.parquet'
    scores = score_responses(rows, client=get_judge_client(judge_key), scale=scale,
                             domain=spec.judge_domain, judge_key=judge_key,
                             resume_path=path.with_suffix('.ckpt.jsonl') if resume else None,
                             max_workers=max_workers)
    data.write_table(scores, path)
    record(path.with_suffix(".manifest.json"), stage="judge",
           inputs=sorted((outputs / "responses" / case_study).glob("*.parquet")),
           specification={"judge": judge_key, "model_id": registry.judge(judge_key).model_id,
                          "domain": spec.judge_domain, "domain_confirmed": spec.judge_domain_confirmed,
                          "scale": scale})
    if scores[schema.SCORE].isna().any():
        raise RuntimeError(f'{path}: incomplete scoring; use --resume to retry failed rows')
    return path
