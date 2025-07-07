# main.py

import json
import logging
from pathlib import Path
from openrouter_client import OpenRouterClient
from ocr_module.ocr import OCRProcessor
from llm_parser.pennypet_processor import pennypet_processor

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OCR and LLM clients
ocr = OCRProcessor(lang="fra")
client_qwen    = OpenRouterClient(model_key="primary")
client_mistral = OpenRouterClient(model_key="secondary")

def process_facture_pennypet(
    file_bytes: bytes,
    formule_client: str,
    llm_provider: str = "qwen"
) -> dict:
    """
    Pipeline complet PennyPet :
    1. OCR (PDF, JPG, PNG) → texte_ocr
    2. Identification des actes via regex
    3. Extraction LLM (Qwen ou Mistral) → lignes + montant_total
    4. Sélection de l'AMV
    5. Calcul du remboursement
    """
    # 1. OCR
    # Détecte le type par magic bytes (PDF vs image)
    try:
        # Tente PDF d'abord
        texte_ocr = ocr.extract_text_from_pdf_bytes(file_bytes)
    except Exception:
        # Sinon traite comme image
        texte_ocr = ocr.extract_text_from_image_bytes(file_bytes)
    logging.info("Texte OCR extrait (%d caractères)", len(texte_ocr))

    # 2. Identification des actes
    actes_detectes = pennypet_processor.identifier_actes_sur_facture(texte_ocr)
    logging.info("Actes détectés : %d", len(actes_detectes))

    # 3. Choix du client LLM
    client = client_qwen if llm_provider.lower() == "qwen" else client_mistral
    logging.info("LLM provider : %s", llm_provider)

    # 4. Extraction via LLM
    messages = [
        {"role":"system","content":(
            "Vous êtes un assistant expert en factures vétérinaires. "
            "Extrayez en JSON un tableau 'lignes' (avec 'animal_uid', 'montant_ht', 'description') "
            "et une clé 'montant_total' (float)."
        )},
        {"role":"user","content":texte_ocr}
    ]
    try:
        response = client.chat(messages)
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error("Erreur extraction LLM : %s", e)
        raise RuntimeError(f"Extraction LLM échouée : {e}")

    # 5. Validation de l'extraction
    lignes = data.get("lignes")
    if not isinstance(lignes, list) or any("montant_ht" not in l for l in lignes):
        raise ValueError("Lignes extraites invalides ou manquantes")
    montant_total = data.get("montant_total")
    if montant_total is None:
        raise ValueError("Champ 'montant_total' manquant")

    # 6. Sélection de l’AMV (plus élevée détectée)
    amv_list = [a["amv"] for a in actes_detectes if "amv" in a]
    amv_detectee = max(amv_list) if amv_list else 1
    logging.info("AMV sélectionnée : %d", amv_detectee)

    # 7. Calcul du remboursement
    formule = formule_client.strip().upper()
    remboursement = pennypet_processor.calculer_remboursement_pennypet(
        montant=montant_total,
        amv=amv_detectee,
        formule=formule
    )

    # 8. Résultat complet
    return {
        "texte_ocr":             texte_ocr,
        "actes_detectes":        actes_detectes,
        "extraction_facture":     data,
        "montant_total":         montant_total,
        "amv_detectee":          amv_detectee,
        "remboursement_pennypet": remboursement
    }

if __name__ == "__main__":
    # Exemple local : lire un fichier PDF ou image
    sample_path = Path("samples/facture_exemple.pdf")
    with open(sample_path, "rb") as f:
        file_bytes = f.read()
    result = process_facture_pennypet(
        file_bytes=file_bytes,
        formule_client="INTEGRAL",
        llm_provider="qwen"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
