"""Thin CLI over the pipeline and scientific analysis services."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from parallax_audit import registry, schema


def setup_logging(verbose: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-7s %(name)s: %(message)s', datefmt='%H:%M:%S'))
    root.addHandler(handler)


def _cases(value):
    if value == 'all':
        return sorted(registry.case_studies())
    registry.case_study(value)
    return [value]


def _models(value):
    if value == 'all-eastern':
        return [key for key, model in registry.models().items() if model.provider == 'openrouter']
    registry.model(value)
    return [value]


def _single_out(args, count):
    if getattr(args, 'out', None) and count != 1:
        raise ValueError('--out accepts exactly one case/model/encoder/scale; use --outputs for batches')
    return Path(args.out) if getattr(args, 'out', None) else None


def dispatch(args):
    registry.validate_configuration()
    outputs = Path(args.outputs)
    if args.command == 'ingest':
        from parallax_audit.data import run_ingest
        print(run_ingest(root=Path(args.source), outputs=Path(args.out) if args.out else outputs))
        return 0
    if args.command == 'report':
        from parallax_audit.report import build
        print(build(outputs=outputs, tables=args.tables, figures=args.figures, B=args.resamples, seed=args.seed))
        return 0
    if args.command == 'archive-run':
        from parallax_audit.pipeline import archive_run
        print(archive_run(source=Path(args.source), outputs=outputs, resamples=args.resamples,
                          seed=args.seed, figures=not args.no_figures))
        return 0
    cases = _cases(args.case_study)
    if args.command == 'query':
        models = _models(args.model)
        out = _single_out(args, len(cases) * len(models))
        for model in models:
            for cs in cases:
                if args.dry_run:
                    spec = registry.model(model)
                    print(f'[dry-run] query model={model} provider={spec.provider} model_id={spec.model_id} '
                          f'case_study={cs} estimated_calls={registry.case_study(cs).n_questions}')
                else:
                    from parallax_audit.pipeline import collect
                    print(collect(model, cs, outputs=outputs, out=out, resume=args.resume,
                                  max_workers=args.max_workers))
    elif args.command == 'embed':
        if bool(args.encoder) == bool(args.all_encoders):
            raise ValueError('embed requires exactly one of --encoder KEY or --all-encoders')
        encoders = list(registry.encoders()) if args.all_encoders else [args.encoder]
        for key in encoders:
            registry.encoder(key)
        out = _single_out(args, len(cases) * len(encoders))
        from parallax_audit.pipeline import embed
        for cs in cases:
            for encoder in encoders:
                print(embed(encoder, cs, outputs=outputs, out=out, device=args.device, batch_size=args.batch_size))
    elif args.command == 'judge':
        registry.judge(args.judge)
        if bool(args.all_scales) == (args.scale is not None):
            raise ValueError('judge requires exactly one of --scale N or --all-scales')
        scales = list(schema.VALID_SCALES) if args.all_scales else [args.scale]
        if any(scale not in schema.VALID_SCALES for scale in scales):
            raise ValueError(f'unknown scale; supported: {schema.VALID_SCALES}')
        out = _single_out(args, len(cases) * len(scales))
        for cs in cases:
            for scale in scales:
                if args.dry_run:
                    print(f'[dry-run] judge judge={args.judge} case_study={cs} scale={scale} '
                          f'estimated_calls<={registry.case_study(cs).n_questions * len(registry.models())} '
                          f'domain_confirmed={registry.case_study(cs).judge_domain_confirmed}')
                else:
                    from parallax_audit.pipeline import judge
                    print(judge(args.judge, cs, scale, outputs=outputs, out=out, resume=args.resume,
                                max_workers=args.max_workers, allow_unconfirmed_domain=args.allow_unconfirmed_domain))
    elif args.command == 'test':
        from parallax_audit import analysis
        for cs in cases:
            kwargs = dict(pool=args.pool, unit=args.unit, permute=args.permute, outputs=outputs)
            if args.method == 'embedding':
                result = analysis.embedding_analysis(cs, args.encoder or 'instructor', **kwargs)
            else:
                result = analysis.judge_analysis(cs, args.judge or 'gpt52', args.scale or 10,
                                                  B=args.resamples, seed=args.seed, **kwargs)
            print(result.to_string(index=False))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog='parallax-audit', description='Comparative black-box auditing of LLM behavior.')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('ingest', 'query', 'embed', 'judge', 'test', 'report', 'archive-run'):
        command = sub.add_parser(name)
        command.add_argument('--outputs', default='outputs', help='canonical artifact root (default: outputs)')
        command.add_argument('--verbose', action='store_true')
        if name in ('query', 'embed', 'judge', 'test'):
            command.add_argument('--case-study', required=True, help='cs1, cs2, cs3, or all')
        if name in ('ingest', 'query', 'embed', 'judge'):
            command.add_argument('--out', default=None, help='single artifact override; ingest: output root')
        if name == 'ingest':
            command.add_argument('--source', default='datasets', help='archived corpus root')
        if name == 'archive-run':
            command.add_argument('--source', default='datasets', help='archived corpus root')
            command.add_argument('--resamples', type=int, default=20000)
            command.add_argument('--seed', type=int, default=0)
            command.add_argument('--no-figures', action='store_true', help='write tables only')
        if name == 'query':
            command.add_argument('--model', required=True)
        if name in ('query', 'judge'):
            command.add_argument('--dry-run', action='store_true', help='plan without constructing clients')
            command.add_argument('--resume', action='store_true', help='resume a matching run and retry failed rows')
            command.add_argument('--max-workers', type=int, default=4)
        if name == 'embed':
            command.add_argument('--encoder')
            command.add_argument('--all-encoders', action='store_true')
            command.add_argument('--device', default=None, help='cpu or cuda; automatic when omitted')
            command.add_argument('--batch-size', type=int, default=64)
        if name == 'judge':
            command.add_argument('--judge', required=True)
            command.add_argument('--scale', type=int)
            command.add_argument('--all-scales', action='store_true')
            command.add_argument('--allow-unconfirmed-domain', action='store_true')
        if name == 'test':
            command.add_argument('--method', choices=['embedding', 'judge'], required=True)
            command.add_argument('--encoder')
            command.add_argument('--judge')
            command.add_argument('--scale', type=int)
            command.add_argument('--pool', default=None, help='default: original eight-model experiment')
            command.add_argument('--unit', choices=['question', 'topic'], default='question')
            command.add_argument('--permute', action='store_true', help='swap target within the selected pool')
        if name in ('test', 'report'):
            command.add_argument('--resamples', type=int, default=20000)
            command.add_argument('--seed', type=int, default=0)
        if name == 'report':
            command.add_argument('--tables', action='store_true')
            command.add_argument('--figures', action='store_true')
    return parser


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    setup_logging(args.verbose)
    try:
        return dispatch(args)
    except (KeyError, ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'error: {exc.args[0] if isinstance(exc, KeyError) else exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
