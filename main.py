import json
import logging
from pathlib import Path
from openrouter_client import OpenRouterClient
from llm_parser.pennypet_processor import PennyPetProcessor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_sample_path() -> Path:
    """
    Retourne le chemin absolu du fichier d'exemple.
    Soulève une erreur explicite si le fichier est absent.
    """
    sample_path = Path(__file__).parent / "samples" / "facture_exemple.pdf"
    if not sample_path.exists():
        logging.error(f"Fichier de test introuvable : {sample_path}")
        raise FileNotFoundError(
            f"Copiez un fichier d'exemple nommé 'facture_exemple.pdf' dans {sample_path.parent}/"
        )
    return sample_path

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
    # Initialisation des clients LLM Vision (hors du main pour usage réutilisable)
    client_qwen = OpenRouterClient(model_key="primary")
    client_mistral = OpenRouterClient(model_key="secondary")
    processor = PennyPetProcessor(client_qwen, client_mistral)

    try:
        result = processor.process_facture_pennypet(
            file_bytes=file_bytes,
            formule_client=formule_client,
            llm_provider=llm_provider
        )
        return result
    except ValueError as ve:
        logging.error(f"Erreur de parsing ou de validation JSON : {ve}")
        raise
    except Exception as e:
        logging.error(f"Erreur lors de l'appel LLM Vision ou du calcul : {e}")
        raise

if __name__ == "__main__":
    try:
        sample_path = get_sample_path()
        with open(sample_path, "rb") as f:
            file_bytes = f.read()
        result = process_facture_pennypet(
            file_bytes=file_bytes,
            formule_client="INTEGRAL",
            llm_provider="qwen"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        logging.critical(f"Échec du pipeline PennyPet : {e}")
        exit(1)
