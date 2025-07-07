# tests/conftest.py
import pytest
from pathlib import Path
from config.pennypet_config import PennyPetConfig
from llm_parser.pennypet_processor import PennyPetProcessor

@pytest.fixture(scope="session")
def base_dir(tmp_path_factory):
    # Répertoire temporaire simulant la structure 'config/' si besoin
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def config(base_dir):
    """Instanciation de la configuration pour charger CSV/JSON."""
    return PennyPetConfig(base_dir=base_dir)

@pytest.fixture
def processor(config, mocker):
    """Processor avec OCR et LLM simulés."""
    # Mock de l'OCRProcessor
    mock_ocr = mocker.Mock()
    mock_ocr.extract_text_from_pdf_bytes.return_value = "texte factice"
    # Mock de OpenRouterClient.chat pour renvoyer un JSON valide
    class DummyResponse:
        def __init__(self, content): self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]
    mock_client = mocker.Mock()
    fake_json = '{"lignes":[{"animal_uid":"A1","montant_ht":10.0,"description":"acte"}],"montant_total":10.0}'
    mock_client.chat.return_value = DummyResponse(fake_json)

    return PennyPetProcessor(
        ocr=mock_ocr,
        client_qwen=mock_client,
        client_mistral=mock_client,
        config=config
    )
