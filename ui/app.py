import sys, os
# ajoute la racine du projet (parent de ui/) en première position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import json
from pathlib import Path
from openrouter_client import OpenRouterClient
from ocr_module.ocr import OCRProcessor
from llm_parser.pennypet_processor import PennyPetProcessor

# Initialisation
st.set_page_config(page_title="PennyPet Invoice", layout="wide")
st.title("PennyPet – Analyse de Facture Vétérinaire")

# Sidebar de configuration
with st.sidebar:
    st.header("Paramètres")
    provider = st.selectbox("LLM Provider", ["qwen", "mistral"])
    formule = st.selectbox("Formule client", ["PREMIUM_BASE", "PREMIUM_PLUS", "INTEGRAL_50", "INTEGRAL_PLUS"])
    animal_id = st.text_input("ID Animal", value="")
    age_animal = st.number_input("Âge de l’animal (en années)", min_value=0, max_value=30, value=2)

# Zone de drag & drop
uploaded = st.file_uploader(
    label="Glissez-déposez votre facture (PDF, JPG, PNG)",
    type=["pdf","jpg","jpeg","png"],
    accept_multiple_files=False
)

if uploaded:
    bytes_data = uploaded.read()
    # OCR
    ocr = OCRProcessor(lang="fra")
    try:
        texte_ocr = ocr.extract_text_from_pdf_bytes(bytes_data)
    except Exception:
        texte_ocr = ocr.extract_text_from_image_bytes(bytes_data)
    st.subheader("Texte extrait (OCR)")
    st.text_area("OCR Output", texte_ocr, height=200)

    # LLM + calcul
    processor = PennyPetProcessor(llm_provider=provider)
    result = processor.process_facture(
        texte_ocr=texte_ocr,
        formule=formule
    )

    # Ajout des champs manuels
    result["animal_id"] = animal_id
    result["age_animal"] = age_animal

    # Affichage des résultats
    st.subheader("Actes détectés")
    st.json(result["actes_detectes"])

    st.subheader("Extraction LLM + Montant total")
    st.json({
        "lignes": result["extraction_facture"]["lignes"],
        "montant_total": result["extraction_facture"]["montant_total"]
    })

    st.subheader("Calcul de remboursement")
    st.json(result["remboursement_pennypet"])

    # Export JSON final
    st.subheader("Données exportables")
    st.download_button(
        label="Télécharger les résultats (JSON)",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="resultat_pennypet.json",
        mime="application/json"
    )
