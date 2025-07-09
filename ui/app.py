# app.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import json
from openrouter_client import OpenRouterClient
from llm_parser.pennypet_processor import PennyPetProcessor
from st_supabase_connection import SupabaseConnection

# Configuration de la page
st.set_page_config(page_title="PennyPet Invoice + DB", layout="wide")
st.title("PennyPet – Extraction & Remboursement")

# 1. Connexion à Supabase
conn = st.connection("supabase", type=SupabaseConnection)

# 2. Choix du modèle LLM Vision
provider = st.sidebar.selectbox("Modèle Vision", ["qwen", "mistral"], index=0)

# 3. Upload de la facture
uploaded = st.file_uploader("Déposez votre facture", type=["pdf", "jpg", "png"])

# 4. Possibles formules pour simulation
formules_possibles = ["START", "PREMIUM", "INTEGRAL", "INTEGRAL_PLUS"]

def select_formule():
    return st.sidebar.selectbox("Formule pour simulation", formules_possibles, index=0)

if uploaded:
    # 5. Extraction d'informations client/animal via LLM Vision
    bytes_data = uploaded.read()
    processor = PennyPetProcessor()
    with st.spinner("Extraction des informations de la facture..."):
        try:
            # Extraction initiale (inclut infos client/animal)
            extraction = processor.extract_lignes_from_image(
                bytes_data, formule="", llm_provider=provider
            )
        except Exception as e:
            st.error(f"Erreur lors de l'extraction : {e}")
            st.stop()

    infos_client = extraction.get("informations_client", {})
    identification = infos_client.get("identification")
    nom_proprietaire = infos_client.get("nom_proprietaire")
    nom_animal = infos_client.get("nom_animal")

    # 6. Recherche automatique dans la base
    res = []
    if identification:
        q = """
        SELECT proprietaire, animal, type_animal, date_naissance, identification, formule
          FROM contrats_animaux
         WHERE identification = :id
         LIMIT 1;
        """
        res = conn.query(q, {"id": identification}).execute().data
    elif nom_proprietaire and nom_animal:
        q = """
        SELECT proprietaire, animal, type_animal, date_naissance, identification, formule
          FROM contrats_animaux
         WHERE proprietaire ILIKE :proprio AND animal ILIKE :animal
         LIMIT 5;
        """
        res = conn.query(q, {"proprio": f"%{nom_proprietaire}%", "animal": f"%{nom_animal}%"}).execute().data

    # 7. Sélection du contrat ou simulation
    if res and len(res) == 1:
        client = res[0]
        st.sidebar.markdown(f"**Propriétaire :** {client['proprietaire']}")
        st.sidebar.markdown(f"**Animal :** {client['animal']} ({client['type_animal']})")
        st.sidebar.markdown(f"**Formule :** {client['formule']}")
    elif res and len(res) > 1:
        choix = st.selectbox(
            "Plusieurs contrats trouvés, sélectionnez le bon :",
            [f"{r['proprietaire']} - {r['animal']} ({r['identification']})" for r in res],
        )
        client = res[[f"{r['proprietaire']} - {r['animal']} ({r['identification']})" for r in res].index(choix)]
        st.sidebar.markdown(f"**Propriétaire :** {client['proprietaire']}")
        st.sidebar.markdown(f"**Animal :** {client['animal']} ({client['type_animal']})")
        st.sidebar.markdown(f"**Formule :** {client['formule']}")
    else:
        st.warning("Aucun contrat trouvé en base pour ce client/animal. Sélectionnez une formule pour simuler la prise en charge.")
        client = {
            "proprietaire": nom_proprietaire or "Simulation",
            "animal": nom_animal or "Simulation",
            "type_animal": "",
            "formule": select_formule()
        }

    # 8. Calcul via LLM Vision et affichage des résultats
    with st.spinner("Analyse et calcul du remboursement..."):
        try:
            result = processor.process_facture_pennypet(
                file_bytes=bytes_data,
                formule_client=client["formule"],
                llm_provider=provider
            )
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
            st.stop()

    st.subheader("Détails du remboursement")
    st.json({
        "lignes":          result["remboursements"],
        "total_remboursé": result["total_remboursement"],
        "reste_à_charge":  result["reste_total_a_charge"]
    })

    # 9. Enregistrement optionnel (si contrat réel)
    if res and st.button("Enregistrer le remboursement"):
        insert_q = """
        INSERT INTO remboursements (
          id_contrat, date_acte, montant_facture, montant_rembourse, reste_a_charge
        ) VALUES (
          (SELECT id FROM contrats_animaux WHERE identification=:id),
          NOW(), :facture, :rembourse, :reste
        );
        """
        conn.query(insert_q, {
          "id":        client["identification"],
          "facture":   result["total_facture"],
          "rembourse": result["total_remboursement"],
          "reste":     result["reste_total_a_charge"]
        }).execute()
        st.success("Opération enregistrée en base.")
else:
    st.info("Importez une facture pour démarrer l’analyse automatique et la simulation de remboursement.")
