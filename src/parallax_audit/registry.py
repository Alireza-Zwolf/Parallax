"""Typed access to the YAML configuration.

Everything the pipeline needs to know about a model, judge, encoder, case study
or baseline pool is declared in ``parallax_audit/config/*.yaml`` and reached through
this module. Nothing else in the package should open those files.

The registry also owns *alias resolution*: the legacy corpus spells the same
eight models sixteen different ways across filenames and dataframe columns
(``deepSeekAWS`` vs ``deepseekAWS``, ``MistralLarge_CS1_Judged_scale4.csv`` vs
``MistralLarge_CS2_scale10.csv``, ...). :func:`resolve_model` maps any of them
onto one canonical key, and raises rather than guessing when a name is
ambiguous -- silently dropping a model is how the original notebooks produced
eight-model plots from seven files.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

CONFIG_DIR = Path(__file__).parent / "config"

Origin = Literal["western", "eastern"]
InstructionStyle = Literal["instructor", "none"]


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    key: str
    display: str
    provider: str
    origin: Origin
    model_id: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Judge:
    key: str
    display: str
    provider: str
    model_id: str


@dataclass(frozen=True)
class Encoder:
    key: str
    display: str
    model_id: str
    dim: int
    instruction_style: InstructionStyle
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseStudy:
    key: str
    display: str
    short: str
    target: str
    n_questions: int
    judge_domain: str
    judge_domain_confirmed: bool
    embed_instruction: str
    legacy_dir: str


@dataclass(frozen=True)
class Pool:
    key: str
    display: str
    description: str
    members: tuple[str, ...]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _read(name: str) -> dict:
    with (CONFIG_DIR / name).open() as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def models() -> dict[str, Model]:
    """Canonical key -> :class:`Model`, in declaration order."""
    return {
        m["key"]: Model(
            key=m["key"],
            display=m["display"],
            provider=m["provider"],
            origin=m["origin"],
            model_id=m["model_id"],
            aliases=tuple(m.get("aliases", ())),
        )
        for m in _read("models.yaml")["models"]
    }


@functools.lru_cache(maxsize=1)
def judges() -> dict[str, Judge]:
    return {j["key"]: Judge(**j) for j in _read("models.yaml")["judges"]}


@functools.lru_cache(maxsize=1)
def encoders() -> dict[str, Encoder]:
    return {
        e["key"]: Encoder(
            key=e["key"],
            display=e["display"],
            model_id=e["model_id"],
            dim=e["dim"],
            instruction_style=e["instruction_style"],
            aliases=tuple(e.get("aliases", ())),
        )
        for e in _read("models.yaml")["encoders"]
    }


@functools.lru_cache(maxsize=1)
def case_studies() -> dict[str, CaseStudy]:
    return {
        c["key"]: CaseStudy(
            key=c["key"],
            display=c["display"],
            short=c["short"],
            target=c["target"],
            n_questions=c["n_questions"],
            judge_domain=c["judge_domain"],
            judge_domain_confirmed=c["judge_domain_confirmed"],
            embed_instruction=c["embed_instruction"],
            legacy_dir=c["legacy_dir"],
        )
        for c in _read("case_studies.yaml")["case_studies"]
    }


@functools.lru_cache(maxsize=1)
def pools() -> dict[str, Pool]:
    raw = _read("pools.yaml")
    return {
        key: Pool(
            key=key,
            display=spec["display"],
            description=spec["description"],
            members=tuple(spec["members"]),
        )
        for key, spec in raw["pools"].items()
    }


@functools.lru_cache(maxsize=1)
def pool_growth() -> dict:
    """Configuration block for the exhaustive pool-growth sweep."""
    return dict(_read("pools.yaml")["pool_growth"])


# --------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------


def model(key: str) -> Model:
    try:
        return models()[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(models())}") from None


def judge(key: str) -> Judge:
    try:
        return judges()[key]
    except KeyError:
        raise KeyError(f"unknown judge {key!r}; known: {sorted(judges())}") from None


def encoder(key: str) -> Encoder:
    try:
        return encoders()[key]
    except KeyError:
        raise KeyError(f"unknown encoder {key!r}; known: {sorted(encoders())}") from None


def case_study(key: str) -> CaseStudy:
    try:
        return case_studies()[key]
    except KeyError:
        raise KeyError(
            f"unknown case study {key!r}; known: {sorted(case_studies())}"
        ) from None


def pool(key: str) -> Pool:
    try:
        return pools()[key]
    except KeyError:
        raise KeyError(f"unknown pool {key!r}; known: {sorted(pools())}") from None


def original_eight() -> tuple[str, ...]:
    """The eight models audited in the submitted paper, in figure order."""
    return (
        "deepseek_r1",
        "cohere_command_r_plus",
        "llama4_maverick",
        "claude_37_sonnet",
        "deepseek_aws",
        "jamba_15_large",
        "meta_ai_chat",
        "mistral_large",
    )


# --------------------------------------------------------------------------
# Alias resolution
# --------------------------------------------------------------------------


def _resolve(text: str, table: dict, kind: str) -> str:
    """Longest-alias-wins substring match of ``text`` against ``table``.

    Longest-first matching is essential: ``deepseek_aws``'s alias
    ``deepseekAWS`` must win over ``deepseek_r1``'s ``deepseek_CS``, and
    ``claude_37_sonnet``'s ``ClaudeSonnet37`` over its own shorter ``claude``.
    """
    haystack = text.lower()
    candidates: list[tuple[int, str]] = []
    for key, record in table.items():
        for alias in (key, *record.aliases):
            if alias.lower() in haystack:
                candidates.append((len(alias), key))

    if not candidates:
        raise ValueError(f"no {kind} alias matches {text!r}")

    best = max(c[0] for c in candidates)
    winners = {key for length, key in candidates if length == best}
    if len(winners) > 1:
        raise ValueError(
            f"ambiguous {kind} for {text!r}: {sorted(winners)} tie at alias "
            f"length {best}; disambiguate in config/models.yaml"
        )
    return winners.pop()


def resolve_model(text: str) -> str:
    """Canonical model key for any legacy filename, path or column name.

    >>> resolve_model("MistralLarge_CS1_Judged_scale4.csv")
    'mistral_large'
    >>> resolve_model("datasets/.../deepSeekAWS/deepseekAWS_CS1_judged_scale7.csv")
    'deepseek_aws'
    """
    return _resolve(text, models(), "model")


def resolve_encoder(text: str) -> str:
    """Canonical encoder key for a legacy directory name, e.g. ``BGE_large``."""
    return _resolve(text, encoders(), "encoder")


@functools.lru_cache(maxsize=1)
def validate_configuration() -> None:
    """Reject duplicate keys and dangling experiment references before execution."""
    for filename, sections in [("models.yaml", ("models", "judges", "encoders")),
                               ("case_studies.yaml", ("case_studies",))]:
        raw = _read(filename)
        for section in sections:
            entries = raw[section]
            keys = [entry["key"] for entry in entries]
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate keys in {filename}/{section}")
    model_keys = set(models())
    for spec in models().values():
        if spec.provider not in {"bedrock", "deepseek", "openrouter", "metaai"}:
            raise ValueError(f"unsupported provider for {spec.key}: {spec.provider}")
    for spec in encoders().values():
        if spec.dim <= 0 or spec.instruction_style not in {"none", "instructor"}:
            raise ValueError(f"invalid encoder configuration: {spec.key}")
    for spec in case_studies().values():
        if spec.target not in model_keys or spec.n_questions < 1:
            raise ValueError(f"invalid case study: {spec.key}")
    for spec in pools().values():
        if len(set(spec.members)) != len(spec.members) or set(spec.members) - model_keys:
            raise ValueError(f"invalid pool members: {spec.key}")
