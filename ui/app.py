import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from st_supabase_connection import SupabaseConnection
from llm_parser.pennypet_processor import PennyPetProcessor

st.set_page_config(page_title="PennyPet Invoice + DB", layout="wide")
st.title("PennyPet – Extraction & Remboursement")

# 1. Connexion à Supabase
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.error(f"Erreur de connexion à Supabase : {e}")
    st.stop()

# 2. Choix du modèle LLM Vision
provider = st.sidebar.selectbox("Modèle Vision", ["qwen", "mistral"], index=0)
formules_possibles = ["START", "PREMIUM", "INTEGRAL", "INTEGRAL_PLUS"]
def select_formule():
    return st.sidebar.selectbox("Formule pour simulation", formules_possibles, index=0)

# 3. Upload de la facture
uploaded = st.file_uploader("Déposez votre facture", type=["pdf", "jpg", "png"])

if uploaded:
    bytes_data = uploaded.read()
    if not bytes_data:
        st.error("Le fichier uploadé est vide ou corrompu.")
        st.stop()

    processor = PennyPetProcessor()

    # 4. Traitement initial pour extraire les infos de base
    with st.spinner("Extraction des informations client en cours..."):
        try:
            # Premier appel pour extraire les infos client uniquement
            result_temp = processor.process_facture_pennypet(
                file_bytes=bytes_data,
                formule_client="INTEGRAL",  # Formule temporaire pour l'extraction
                llm_provider=provider
            )
            if result_temp is None or not isinstance(result_temp, dict):
                st.error("Impossible d'extraire les informations de la facture.")
                st.stop()
        except Exception as e:
            st.error(f"Erreur lors de l'extraction des informations : {e}")
            st.stop()

    # 5. Récupération des infos client/animal extraites
    infos = result_temp.get("infos_client", {}) if isinstance(result_temp, dict) else {}
    identification = infos.get("identification")
    nom_proprietaire = infos.get("nom_proprietaire")
    nom_animal = infos.get("nom_animal")

    # 6. Recherche automatique dans la base via st-supabase-connection
    res = []
    try:
        if identification:
            res = (
                conn
                .table("contrats_animaux")
                .select("proprietaire,animal,type_animal,date_naissance,identification,formule")
                .eq("identification", identification)
                .limit(1)
                .execute()
                .data
            )
        elif nom_proprietaire and nom_animal:
            res = (
                conn
                .table("contrats_animaux")
                .select("proprietaire,animal,type_animal,date_naissance,identification,formule")
                .ilike("proprietaire", f"%{nom_proprietaire}%")
                .ilike("animal", f"%{nom_animal}%")
                .limit(5)
                .execute()
                .data
            )
    except Exception as e:
        st.warning(f"Recherche Supabase échouée : {e}")

    # 7. Sélection du contrat ou simulation
    if res and len(res) == 1:
        client = res[0]
        formule_client = client["formule"]  # Récupération de la formule réelle
    elif res and len(res) > 1:
        choix = st.selectbox(
            "Plusieurs contrats trouvés, sélectionnez le bon :",
            [f"{r['proprietaire']} - {r['animal']} ({r['identification']})" for r in res]
        )
        idx = [f"{r['proprietaire']} - {r['animal']} ({r['identification']})" for r in res].index(choix)
        client = res[idx]
        formule_client = client["formule"]  # Récupération de la formule réelle
    else:
        st.warning("Aucun contrat trouvé. Sélectionnez une formule pour simuler la prise en charge.")
        formule_client = select_formule()  # Formule manuelle sélectionnée
        client = {
            "proprietaire": nom_proprietaire or "Simulation",
            "animal": nom_animal or "Simulation",
            "type_animal": "",
            "formule": formule_client
        }

    # 8. Affichage des infos client/animal
    st.sidebar.markdown(f"**Propriétaire :** {client.get('proprietaire','')}")
    st.sidebar.markdown(f"**Animal :** {client.get('animal','')} ({client.get('type_animal','')})")
    st.sidebar.markdown(f"**Formule :** {client.get('formule','')}")

    # 9. Traitement complet avec la bonne formule
    with st.spinner("Analyse et calcul du remboursement en cours..."):
        try:
            result = processor.process_facture_pennypet(
                file_bytes=bytes_data,
                formule_client=formule_client,  # ← CORRECTION PRINCIPALE : formule non vide
                llm_provider=provider
            )
            if result is None or not isinstance(result, dict):
                st.error("Le traitement n'a retourné aucun résultat exploitable.")
                st.stop()
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
            st.stop()

    # 10. Affichage du résultat de remboursement
    st.subheader("Détails du remboursement")
    try:
        st.json({
            "lignes":          result["remboursements"],
            "total_facture":   result["total_facture"],
            "total_remboursé": result["total_remboursement"],
            "reste_à_charge":  result["reste_total_a_charge"]
        })
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du résultat : {e}")

    # 11. Enregistrement optionnel (si contrat réel)
    if res and st.button("Enregistrer le remboursement"):
        try:
            contrat_id = (
                conn
                .table("contrats_animaux")
                .select("id")
                .eq("identification", client["identification"])
                .limit(1)
                .execute()
                .data[0]["id"]
            )
            _ = (
                conn
                .table("remboursements")
                .insert([{
                    "id_contrat":       contrat_id,
                    "date_acte":        "now()",
                    "montant_facture":  result["total_facture"],
                    "montant_rembourse": result["total_remboursement"],
                    "reste_a_charge":   result["reste_total_a_charge"]
                }])
                .execute()
            )
            st.success("Opération enregistrée en base.")
        except Exception as e:
            st.error(f"Erreur lors de l'enregistrement : {e}")
else:
    st.info("Importez une facture pour démarrer l'analyse automatique et la simulation de remboursement.")
