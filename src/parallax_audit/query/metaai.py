"""Meta AI provider.

Serves ``meta_ai_chat`` (``provider: metaai`` in models.yaml) via the
unofficial ``meta_ai_api`` package, which scrapes the Meta AI chat endpoint
rather than calling a documented API -- there is no ``model_id`` to pass.
"""

from __future__ import annotations


class MetaAIProvider:
    """Provider wrapping the ``meta_ai_api.MetaAI`` client."""

    def __init__(self) -> None:
        try:
            from meta_ai_api import MetaAI
        except ImportError as exc:
            raise RuntimeError(
                "the 'meta_ai_api' package is required for the metaai provider; "
                "install it with `pip install meta-ai-api`"
            ) from exc

        self._client_factory = MetaAI

    def generate(self, prompt: str) -> str:
        """Send ``prompt``; raises on API failure."""
        # Each question starts an independent session, including concurrent calls.
        response = self._client_factory().prompt(message=prompt)
        return response["message"]
