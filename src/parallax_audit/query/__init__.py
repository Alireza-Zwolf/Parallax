"""Response collection: turn a set of questions into model responses.

One abstraction: a :class:`~parallax_audit.query.base.Provider`
protocol implemented once per backend (:mod:`parallax_audit.query.bedrock`,
:mod:`parallax_audit.query.deepseek`, :mod:`parallax_audit.query.metaai`,
:mod:`parallax_audit.query.openrouter`), and a single
:func:`~parallax_audit.query.base.collect_responses` loop shared by all of them.

:func:`~parallax_audit.query.factory.get_provider` dispatches on
``registry.model(key).provider`` to build the right client. Provider modules
import their SDKs lazily so that ``import parallax_audit.query`` never requires
``boto3``, ``openai`` or ``meta_ai_api`` to be installed, and so that a
``--dry-run`` can be proven never to construct a real client.
"""

from __future__ import annotations

from parallax_audit.query.base import Provider, collect_responses
from parallax_audit.query.factory import get_provider

__all__ = ["Provider", "collect_responses", "get_provider"]
