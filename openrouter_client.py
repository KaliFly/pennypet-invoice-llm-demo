# openrouter_client.py
import os
from dotenv import load_dotenv
import openai

load_dotenv()

class OpenRouterClient:
    def __init__(self, model_key: str):
        if model_key == "primary":
            api_key = os.getenv("OPENROUTER_API_KEY_QWEN")
            self.model = os.getenv("MODEL_PRIMARY")
        else:
            api_key = os.getenv("OPENROUTER_API_KEY_MISTRAL")
            self.model = os.getenv("MODEL_SECONDARY")

        if not api_key:
            raise ValueError(f"Clé API manquante pour le modèle {model_key!r}")

        # Instancie le client OpenAI v1.x pointant vers OpenRouter
        self.client = openai.Client(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def chat(self, messages, temperature=0.1, max_tokens=4000):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response
