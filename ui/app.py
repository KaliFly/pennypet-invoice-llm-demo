# app.py
# Permet d’importer modules racine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os
import streamlit as st
import json
from openrouter_client import OpenRouterClient
from ocr_module.ocr import OCRProcessor
from llm_parser.pennypet_processor import PennyPetProcessor
from st_supabase_connection import SupabaseConnection


# Configuration de la page
st.set_page_config(page_title="PennyPet Invoice + DB", layout="wide")
st.title("PennyPet – Extraction & Remboursement")

# 1. Connexion à Supabase
conn = st.connection("supabase", type=SupabaseConnection)

# 2. Sélection du modèle IA
provider = st.sidebar.selectbox("Modèle IA", ["qwen", "mistral"], index=0)

# 3. Upload de la facture et saisie de l’ID animal
uploaded = st.file_uploader("Déposez votre facture", type=["pdf","jpg","png"])
animal_id = st.sidebar.text_input("ID Animal")

if uploaded and animal_id:
    # 4. Recherche du client en base
    q = """
    SELECT proprietaire, animal, type_animal, date_naissance, identification, formule
      FROM contrats_animaux
     WHERE identification = :id
     LIMIT 1;
    """
    res = conn.query(q, {"id": animal_id}).execute().data
    if not res:
        st.error("Aucun contrat trouvé pour cet ID animal.")
        st.stop()
    client = res[0]
    st.sidebar.markdown(f"**Propriétaire :** {client['proprietaire']}")
    st.sidebar.markdown(f"**Animal :** {client['animal']} ({client['type_animal']})")
    st.sidebar.markdown(f"**Formule :** {client['formule']}")

    # 5. Lecture OCR
    bytes_data = uploaded.read()
    ocr = OCRProcessor(lang="fra")
    try:
        texte = ocr.extract_text_from_pdf_bytes(bytes_data)
    except Exception:
        texte = ocr.extract_text_from_image_bytes(bytes_data)

    st.subheader("Texte extrait (OCR)")
    st.text_area("OCR Output", texte, height=200)

    # 6. Calcul de remboursement via le processor
    processor = PennyPetProcessor(llm_provider=provider)
    result = processor.process_facture_pennypet(
        file_bytes=bytes_data,
        formule_client=client["formule"],
        llm_provider=provider
    )

    # 7. Affichage des détails
    st.subheader("Détails du remboursement")
    st.json({
        "lignes":          result["remboursements"],
        "total_remboursé": result["total_remboursement"],
        "reste_à_charge":  result["reste_total_a_charge"]
    })

    # 8. Optionnel : enregistrement de l’opération
    if st.button("Enregistrer le remboursement"):
        insert_q = """
        INSERT INTO remboursements (
          id_contrat, date_acte, montant_facture, montant_rembourse, reste_a_charge
        ) VALUES (
          (SELECT id FROM contrats_animaux WHERE identification=:id),
          NOW(), :facture, :rembourse, :reste
        );
        """
        conn.query(insert_q, {
          "id":          animal_id,
          "facture":     result["total_facture"],
          "rembourse":   result["total_remboursement"],
          "reste":       result["reste_total_a_charge"]
        }).execute()
        st.success("Opération enregistrée en base.")

else:
    st.info("Importez une facture et renseignez l’ID animal pour démarrer.")
