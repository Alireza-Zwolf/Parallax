"""Turning responses into ordinal deviation signals.

Two independent measurement methods, per the paper's Section 2.2:

    :mod:`parallax_audit.score.embed`   Embedding-Transformation (GPU, no LLM calls)
    :mod:`parallax_audit.score.judge`   LLM-as-a-Judge (paid API calls)
    :mod:`parallax_audit.score.rubrics` the judge prompt templates shared by both
                                  judge providers

Submodules are imported lazily from here: :mod:`parallax_audit.score.embed` pulls in
``torch`` and ``sentence-transformers``, and :mod:`parallax_audit.score.judge`'s
concrete clients pull in the ``openai`` / ``google-generativeai`` SDKs (lazily,
inside their own ``__init__``), so ``import parallax_audit.score`` stays cheap and
``import parallax_audit.score.rubrics`` alone requires neither.
"""

from __future__ import annotations

__all__: list[str] = []
