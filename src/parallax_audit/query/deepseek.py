"""DeepSeek provider.

Serves the ``deepseek_r1`` model (``provider: deepseek`` in models.yaml)
against DeepSeek's own OpenAI-compatible endpoint (distinct from
``deepseek_aws``, which goes through Bedrock). The model id and system prompt
are the ones used for the paper.
"""

from __future__ import annotations

import os

_SYSTEM_PROMPT = "You are a helpful assistant."


class DeepSeekProvider:
    """Provider for DeepSeek's own API (``https://api.deepseek.com``)."""

    def __init__(self, model_id: str) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "the 'openai' package is required for the deepseek provider; "
                "install it with `pip install parallax_audit[query]`"
            ) from exc

        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set; export it or add it to .env "
                "(see .env.example)"
            )
        self._client = openai.OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=60.0, max_retries=0)
        self._model_id = model_id

    def generate(self, prompt: str) -> str:
        """Send ``prompt`` as a single user turn; raises on API failure."""
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return response.choices[0].message.content
