import json
import logging
from pathlib import Path
from openrouter_client import OpenRouterClient
from llm_parser.pennypet_processor import PennyPetProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize LLM Vision clients
client_qwen = OpenRouterClient(model_key="primary")
client_mistral = OpenRouterClient(model_key="secondary")
processor = PennyPetProcessor(client_qwen, client_mistral)

def process_facture_pennypet(
    file_bytes: bytes,
    formule_client: str,
    llm_provider: str = "qwen"
) -> dict:
    """
    Pipeline PennyPet 100% LLM Vision :
    1. Extraction directe via LLM Vision (Qwen ou Mistral) → lignes + montant_total
    2. Calcul du remboursement ligne par ligne selon code_acte et formule
    """
    result = processor.process_facture_pennypet(
        file_bytes=file_bytes,
        formule_client=formule_client,
        llm_provider=llm_provider
    )
    return result

if __name__ == "__main__":
    sample_path = Path("samples/facture_exemple.pdf")
    with open(sample_path, "rb") as f:
        file_bytes = f.read()
    result = process_facture_pennypet(
        file_bytes=file_bytes,
        formule_client="INTEGRAL",
        llm_provider="qwen"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
