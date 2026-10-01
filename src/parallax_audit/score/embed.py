"""GPU embedding runner.

:func:`embed_responses` loads an encoder's weights once (cached across calls,
see ``_load_model``) and batches every text in one
``SentenceTransformer.encode`` invocation. The model runs in ``eval()`` mode,
so output is deterministic for fixed weights and inputs; GPU kernels are not
guaranteed bit-identical across different batch shapes or hardware.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np

from parallax_audit import registry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_id: str, device: str):
    """Load and cache a SentenceTransformer by (model_id, device).

    Cached at module scope (not just "once per call") so that a script
    calling :func:`embed_responses` for several encoders, or repeatedly for
    the same one, never reloads weights it already has resident.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("install parallax_audit[embed] to generate embeddings") from exc
    logger.info("loading encoder %s on %s", model_id, device)
    model = SentenceTransformer(model_id, device=device)
    model.eval()
    return model


def embed_responses(
    texts: Sequence[str],
    encoder_key: str,
    instruction: str | None = None,
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Embed ``texts`` with the encoder registered under ``encoder_key``.

    Returns a float32, L2-normalised array of shape ``(len(texts), dim)``
    where ``dim`` is ``registry.encoder(encoder_key).dim``.

    Two instruction-conditioning paths, selected by
    ``registry.encoder(key).instruction_style``:

    * ``"instructor"`` -- INSTRUCTOR's own scheme, which pairs an
      instruction with each text: ``[[instruction, text], ...]``. The
      instruction is encoded as a preceding segment but excluded from mean
      pooling. This relies on sentence-transformers special-casing the exact
      model name ``hkunlp/instructor-large``: do not load it from a local
      path or alias, or pooling silently includes the prompt tokens.
    * ``"none"`` -- raw text, no instruction prefix. This is how the paper's
      BGE-large / MPNet / MiniLM numbers were produced; an ``instruction``
      argument is ignored with a warning for these encoders.
    """
    if batch_size < 1 or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("embedding requires nonempty texts and a positive batch size")
    encoder = registry.encoder(encoder_key)
    if device is None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("install parallax_audit[embed] to generate embeddings") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(encoder.model_id, device)

    if encoder.instruction_style == "instructor":
        if not instruction:
            raise ValueError(
                f"encoder {encoder_key!r} has instruction_style='instructor' "
                "and requires a non-empty `instruction`"
            )
        inputs: list[str] | list[list[str]] = [[instruction, text] for text in texts]
    elif encoder.instruction_style == "none":
        if instruction is not None:
            logger.warning(
                "instruction given for encoder %r (instruction_style='none'); "
                "ignoring it to match how the published numbers for this "
                "encoder were produced",
                encoder_key,
            )
        inputs = list(texts)
    else:
        raise ValueError(f"unknown instruction_style {encoder.instruction_style!r}")

    # SentenceTransformer.encode manages its own inference/no-grad context.
    vectors = model.encode(
        inputs, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )

    vectors = np.asarray(vectors, dtype=np.float32)
    expected_shape = (len(texts), encoder.dim)
    if vectors.shape != expected_shape:
        raise ValueError(
            f"encoder {encoder_key!r} produced shape {vectors.shape}, "
            f"expected {expected_shape} -- registry.encoder(...).dim is "
            "wrong or the model's output projection changed"
        )
    return vectors
