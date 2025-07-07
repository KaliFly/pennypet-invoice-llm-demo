from typing import Optional
from config.pennypet_config import PennyPetConfig
from ocr_module.ocr import OCRProcessor
from openrouter_client import OpenRouterClient
from llm_parser.parser import InvoiceParser

class PennyPetProcessor:
    def __init__(
        self,
        ocr: OCRProcessor | None = None,
        client_qwen: OpenRouterClient | None = None,
        client_mistral: OpenRouterClient | None = None,
        config: PennyPetConfig | None = None
    ):
        # injection pour les tests, valeurs par défaut sinon
        self.config         = config or PennyPetConfig()
        self.ocr            = ocr or OCRProcessor(lang="fra")
        self.client_qwen    = client_qwen or OpenRouterClient(model_key="primary")
        self.client_mistral = client_mistral or OpenRouterClient(model_key="secondary")
# config/__init__.py
# Ce fichier peut rester vide, il permet de traiter config/ comme un package Python.
