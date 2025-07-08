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
ocr = OCRProcessor(lang="french")
client_qwen = OpenRouterClient(model_key="primary")
client_mistral = OpenRouterClient(model_key="secondary")

def process_facture_pennypet(
    file_bytes: bytes,
    formule_client: str,
    llm_provider: str = "qwen"
) -> dict:
    """
    Pipeline PennyPet sans AMV et sans stockage en base :
    1. OCR → texte_ocr
    2. Identification des actes via regex (optionnel)
    3. Extraction LLM (Qwen ou Mistral) → lignes + montant_total
    4. Calcul du remboursement ligne par ligne selon code_acte et formule
    """
    # 1. OCR
    try:
        texte_ocr = ocr.extract_text_from_pdf_bytes(file_bytes)
    except Exception:
        texte_ocr = ocr.extract_text_from_image_bytes(file_bytes)
    logging.info("Texte OCR extrait (%d caractères)", len(texte_ocr))

    # 2. Identification des actes (optionnel)
    actes_detectes = pennypet_processor.identifier_actes_sur_facture(texte_ocr)
    logging.info("Actes détectés : %d", len(actes_detectes))

    # 3. Choix du client LLM
    client = client_qwen if llm_provider.lower() == "qwen" else client_mistral
    logging.info("LLM provider : %s", llm_provider)

    # 4. Extraction via LLM
    messages = [
        {
            "role": "system",
            "content": (
                "Vous êtes un assistant expert en factures vétérinaires. "
                "Extrayez en JSON un tableau 'lignes' contenant pour chaque ligne : "
                "'animal_uid', 'code_acte', 'montant_ht', 'description', "
                "et une clé 'montant_total' de type float."
            )
        },
        {"role": "user", "content": texte_ocr}
    ]
    try:
        response = client.chat(messages)
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error("Erreur extraction LLM : %s", e)
        raise RuntimeError(f"Extraction LLM échouée : {e}")

    # 5. Validation de l'extraction
    lignes = data.get("lignes")
    if not isinstance(lignes, list) or any(
        not all(key in ligne for key in ("montant_ht", "code_acte")) for ligne in lignes
    ):
        raise ValueError("Lignes extraites invalides ou manquantes")
    montant_total = data.get("montant_total")
    if montant_total is None:
        raise ValueError("Champ 'montant_total' manquant")

    # 6. Calcul du remboursement ligne par ligne (sans AMV)
    formule = formule_client.strip().upper()
    remboursements = []
    for ligne in lignes:
        montant = float(ligne["montant_ht"])
        code_acte = ligne["code_acte"]
        remb = pennypet_processor.calculer_remboursement_pennypet(
            montant=montant,
            code_acte=code_acte,
            formule=formule
        )
        # Fusionner données ligne + calcul
        remboursements.append({**ligne, **remb})

    # 7. Retour du résultat complet
    return {
        "texte_ocr":          texte_ocr,
        "actes_detectes":     actes_detectes,
        "extraction_facture": data,
        "montant_total":      montant_total,
        "formule_utilisee":   formule,
        "remboursements":     remboursements
    }

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
