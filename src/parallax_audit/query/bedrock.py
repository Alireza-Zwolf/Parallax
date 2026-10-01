"""AWS Bedrock provider.

Serves DeepSeek-AWS, Claude 3.7 Sonnet, Cohere Command R+, Llama 4 Maverick,
Jamba 1.5 Large and Mistral Large (``provider: bedrock`` in models.yaml).
The generation settings (``maxTokens=2000, temperature=0.7, topP=0.9`` via
``converse``) are the ones used for the paper and must stay identical so new
models are queried under the same conditions.
"""

from __future__ import annotations

import os

_INFERENCE_CONFIG = {"maxTokens": 2000, "temperature": 0.7, "topP": 0.9}


class BedrockProvider:
    """Provider for models hosted on AWS Bedrock, via the ``converse`` API."""

    def __init__(self, model_id: str) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "the 'boto3' package is required for the bedrock provider; "
                "install it with `pip install parallax_audit[query]`"
            ) from exc

        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id

    def generate(self, prompt: str) -> str:
        """Send ``prompt`` as a single user turn; raises on API failure."""
        response = self._client.converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=_INFERENCE_CONFIG,
        )
        return response["output"]["message"]["content"][0]["text"]
