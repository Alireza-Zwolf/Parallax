"""OpenRouter provider: Qwen3-Max, GLM-5.2 and Kimi-K2.6.

OpenRouter exposes an OpenAI-API-compatible endpoint, so we reuse the
``openai`` SDK pointed at OpenRouter's base URL rather than writing a new
HTTP client. The SDK is imported inside :meth:`OpenRouterProvider.__init__`
(not at module scope) so that ``import parallax_audit.query.openrouter`` succeeds
even when ``openai`` is not installed, and so a ``--dry-run`` that never
constructs this class can never trigger a network call.
"""

from __future__ import annotations

import os


class OpenRouterProvider:
    """Provider for any model served through OpenRouter."""

    def __init__(self, model_id: str) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "the 'openai' package is required for the openrouter provider; "
                "install it with `pip install parallax_audit[query]`"
            ) from exc

        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; export it or add it to .env "
                "(see .env.example)"
            )
        self._client = openai.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=60.0, max_retries=0)
        self._model_id = model_id

    def generate(self, prompt: str) -> str:
        """Send ``prompt`` as a single user turn; raises on API failure."""
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
