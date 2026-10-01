"""Provider factory: dispatch on ``registry.model(key).provider``.

The single entry point the rest of the codebase should use to obtain a
:class:`~parallax_audit.query.base.Provider`. Each branch imports its provider
module lazily so that constructing, say, a Bedrock provider never touches
the ``openai`` SDK, and vice versa.
"""

from __future__ import annotations

from parallax_audit import registry
from parallax_audit.query.base import Provider

def get_provider(model_key: str) -> Provider:
    """Build the :class:`Provider` for ``model_key``.

    Raises ``KeyError`` (from :func:`parallax_audit.registry.model`) for an unknown
    model key, and ``RuntimeError`` -- not a bare ``ImportError`` -- when the
    provider's SDK is not installed or its API key is not configured.
    """
    spec = registry.model(model_key)

    if spec.provider == "openrouter":
        from parallax_audit.query.openrouter import OpenRouterProvider

        return OpenRouterProvider(spec.model_id)
    if spec.provider == "bedrock":
        from parallax_audit.query.bedrock import BedrockProvider

        return BedrockProvider(spec.model_id)
    if spec.provider == "deepseek":
        from parallax_audit.query.deepseek import DeepSeekProvider

        return DeepSeekProvider(spec.model_id)
    if spec.provider == "metaai":
        from parallax_audit.query.metaai import MetaAIProvider

        return MetaAIProvider()

    raise ValueError(f"no query provider registered for {spec.provider!r} (model {model_key!r})")
