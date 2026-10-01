"""Shared retries, resumable checkpoints, and run provenance."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
import subprocess

from parallax_audit.data import write_json


def delay(attempt: int, *, base: float = 1.0, cap: float = 30.0) -> float:
    return min(cap, base * 2**attempt) * (.5 + random.random())


def retryable(error: Exception) -> bool:
    status = getattr(error, 'status_code', None)
    response = getattr(error, 'response', None)
    if isinstance(response, dict):
        status = response.get('ResponseMetadata', {}).get('HTTPStatusCode', status)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    return not isinstance(error, (ValueError, TypeError, ImportError))


def bind(path: Path, frame, specification: dict) -> None:
    """Bind a checkpoint to its exact inputs and settings."""
    columns = list(frame.columns)
    ordered = frame.sort_values(columns, kind='stable', na_position='last')
    fingerprint = hashlib.sha256((ordered.to_json(orient='split', index=False) +
                                  json.dumps(specification, sort_keys=True)).encode()).hexdigest()
    manifest = path.with_suffix(path.suffix + '.manifest.json')
    expected = {'version': 1, 'fingerprint': fingerprint, 'specification': specification}
    path.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError('checkpoint belongs to different inputs or settings; use a new output path')
    elif path.exists() and path.stat().st_size:
        raise ValueError('checkpoint has no run manifest; use a new output path')
    else:
        write_json(expected, manifest)


def read_records(path: Path) -> list[dict]:
    """Read checkpoint records, repairing only a truncated final write."""
    if not path.exists():
        return []
    contents = path.read_bytes()
    records = []
    offset = 0
    lines = contents.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if index == len(lines)-1 and not line.endswith(b'\n'):
                with path.open('r+b') as file:
                    file.truncate(offset)
                break
            raise ValueError(f'corrupt checkpoint record {index+1}: {path}') from exc
        if not isinstance(record, dict):
            raise ValueError(f'checkpoint record must be an object: {path}')
        records.append(record)
        offset += len(line)
    if records and contents and not contents.endswith(b'\n') and path.stat().st_size == len(contents):
        with path.open('ab') as file:
            file.write(b'\n')
    return records


def record(path: Path, *, stage: str, inputs: list[Path], specification=None):
    """Write input hashes, software versions, and Git revision for a run."""
    packages = {}
    for name in ('parallax-audit', 'numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    if is_dataclass(specification):
        specification = asdict(specification)
    write_json({'stage': stage, 'revision': revision, 'dirty': dirty, 'versions': packages,
                'specification': specification,
                'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}, path)
