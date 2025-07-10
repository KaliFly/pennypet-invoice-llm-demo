# openrouter_client.py

import streamlit as st
import openai
import base64
import time
import random
from typing import List, Dict, Any, Optional, Union

class OpenRouterClient:
    """
    Wrapper for the OpenAI Python SDK targeting the OpenRouter.ai API endpoint.
    Supports two providers (primary → Qwen, secondary → Mistral), including vision models.
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

        self.client = openai.Client(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def chat(
        self,
        messages: List[Dict[str, Union[str, list, dict]]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
        stop: Optional[List[str]] = None,
        retries: int = 3
    ) -> Any:
        """
        Send a chat completion request with retries and exponential backoff.

        :param messages: List of {"role": "...", "content": "..."} dicts (content can be str or list for vision)
        :param temperature: Sampling temperature
        :param max_tokens: Maximum number of tokens to generate
        :param stop: Optional list of stop sequences
        :param retries: Number of retry attempts on failure
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

        last_exception = None
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(**params)
                return response
            except Exception as e:
                last_exception = e
                wait_time = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait_time)
        raise RuntimeError(f"OpenRouter API failed after {retries} attempts: {last_exception}")

    def analyze_invoice_image(
        self,
        image_bytes: bytes,
        formule_client: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        retries: int = 3
    ) -> Any:
        """
        Send an invoice image (PDF/JPG/PNG) to a vision LLM for extraction.
        The prompt requests a structured JSON with all relevant fields.
        Includes robust prompt formatting to reduce LLM hallucinations.
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            "Vous êtes un expert en factures vétérinaires.\n"
            "À partir de l'image fournie, extrayez UNIQUEMENT un objet JSON strictement conforme au schéma suivant :\n"
            "{\n"
            '  "texte_ocr": "Texte complet extrait",\n'
            '  "lignes": [\n'
            '    {\n'
            '      "animal_uid": "string (optionnel)",\n'
            '      "code_acte": "string",\n'
            '      "description": "string",\n'
            '      "montant_ht": float\n'
            "    }\n"
            "  ],\n"
            '  "montant_total": float,\n'
            '  "informations_client": {\n'
            '    "nom_proprietaire": "string",\n'
            '    "nom_animal": "string",\n'
            '    "identification": "string"\n'
            "  }\n"
            "}\n"
            "\n"
            f"Formule d'assurance à prendre en compte : {formule_client}\n"
            "RENVOIE UNIQUEMENT DU JSON VALIDE, sans explication, sans balise Markdown, sans texte avant ou après."
        )
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analysez cette facture vétérinaire et extrayez toutes les informations selon le format JSON demandé."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries
        )
