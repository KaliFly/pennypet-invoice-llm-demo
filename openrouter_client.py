# openrouter_client.py

import streamlit as st
import openai
from typing import List, Dict, Any

class OpenRouterClient:
    """
    A thin wrapper around the OpenAI Python SDK that targets the OpenRouter.ai
    API endpoint. Supports two providers (primary → Qwen, secondary → Mistral).
    """

    def __init__(self, model_key: str):
        # Retrieve API key and model from Streamlit secrets
        if model_key == "primary":
            api_key = st.secrets["openrouter"]["API_KEY_QWEN"]
            self.model = st.secrets["openrouter"].get("MODEL_PRIMARY", "qwen/qwen2.5-vl-32b-instruct:free")
        elif model_key == "secondary":
            api_key = st.secrets["openrouter"]["API_KEY_MISTRAL"]
            self.model = st.secrets["openrouter"].get("MODEL_SECONDARY", "mistralai/mistral-small-3.2-24b-instruct:free")
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
