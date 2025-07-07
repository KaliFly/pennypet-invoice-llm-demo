# openrouter_client.py

import os
from dotenv import load_dotenv
import openai
from typing import List, Dict, Any

load_dotenv()


class OpenRouterClient:
    """
    A thin wrapper around the OpenAI Python SDK that targets the OpenRouter.ai
    API endpoint. Supports two providers (primary → Qwen, secondary → Mistral).
    """

    def __init__(self, model_key: str):
        # Determine which API key and model to use
        if model_key == "primary":
            api_key = os.getenv("OPENROUTER_API_KEY_QWEN")
            self.model = os.getenv("MODEL_PRIMARY")
        elif model_key == "secondary":
            api_key = os.getenv("OPENROUTER_API_KEY_MISTRAL")
            self.model = os.getenv("MODEL_SECONDARY")
        else:
            raise ValueError(f"Unknown model_key '{model_key}'. Use 'primary' or 'secondary'.")

        if not api_key:
            raise ValueError(f"Missing API key for model_key={model_key!r}")

        # Instantiate the OpenAI client pointed at OpenRouter
        self.client = openai.Client(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
        stop: List[str] = None
    ) -> Any:
        """
        Send a chat completion request.

        :param messages: List of {"role": "...", "content": "..."} dicts
        :param temperature: Sampling temperature
        :param max_tokens: Maximum number of tokens to generate
        :param stop: Optional list of stop sequences
        :return: The raw response object from OpenAI SDK
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if stop:
            params["stop"] = stop

        response = self.client.chat.completions.create(**params)
        return response
