import sys
import os
import unicodedata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from st_supabase_connection import SupabaseConnection
from llm_parser.pennypet_processor import PennyPetProcessor

# --- Configuration de la page ---
st.set_page_config(page_title="PennyPet Invoice + DB", layout="wide")
st.title("PennyPet – Extraction & Remboursement")

# --- Connexion à Supabase ---
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.error(f"Connexion Supabase impossible : {e}")
    st.stop()

# --- Choix du modèle LLM Vision ---
provider = st.sidebar.selectbox("Modèle Vision", ["qwen", "mistral"], index=0)

# --- Upload de la facture ---
uploaded = st.file_uploader("Déposez votre facture", type=["pdf", "jpg", "png"])

# --- Formules pour simulation ---
formules_possibles = ["START", "PREMIUM", "INTEGRAL", "INTEGRAL_PLUS"]
def select_formule():
    return st.sidebar.selectbox("Formule pour simulation", formules_possibles, index=0)

# --- Fonction de nettoyage et de normalisation ---
def normalize_nom(nom):
    if not nom:
        return ""
    s = nom.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s

if uploaded:
    bytes_data = uploaded.read()
    processor = PennyPetProcessor()

    # --- Extraction initiale (infos client/animal) ---
    with st.spinner("Extraction des informations de la facture..."):
        try:
            extraction, _raw = processor.extract_lignes_from_image(
                bytes_data, formule="", llm_provider=provider
            )
        except ValueError as ve:
            st.error(f"Format de réponse inattendu (JSON ou parsing) : {ve}")
            st.stop()
        except Exception as e:
            st.error(f"Erreur lors de l'extraction (réseau ou API) : {e}")
            st.stop()

    # --- Récupération et nettoyage des infos client/animal extraites ---
    infos_client = extraction.get("informations_client", {}) or {}
    identification = infos_client.get("identification")
    nom_proprietaire = normalize_nom(infos_client.get("nom_proprietaire"))
    nom_animal = normalize_nom(infos_client.get("nom_animal"))

    # --- Affichage debug des valeurs extraites ---
    st.write(f"Propriétaire extrait (normalisé) : [{nom_proprietaire}]")
    st.write(f"Animal extrait (normalisé) : [{nom_animal}]")

    # --- Recherche automatique dans la base ---
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
                .ilike("proprietaire", f"*{nom_proprietaire}*")
                .ilike("animal", f"*{nom_animal}*")
                .limit(5)
                .execute()
                .data
            )
    except Exception as e:
        st.warning(f"Recherche Supabase impossible : {e}")
        res = []

    # --- Affichage debug du résultat Supabase ---
    st.write("Résultat Supabase brut :", res)

    # --- Sélection du contrat ou fallback manuel ---
    if res and len(res) == 1:
        client = res[0]
    elif res and len(res) > 1:
        choix = st.selectbox(
            "Plusieurs contrats trouvés, sélectionnez le bon :",
            [f"{r['proprietaire']} - {r['animal']} ({r['identification']})" for r in res]
        )
        idx = [f"{r['proprietaire']} - {r['animal']} ({r['identification']})" for r in res].index(choix)
        client = res[idx]
    else:
        st.warning("Aucun contrat trouvé automatiquement. Sélectionnez une formule pour simuler la prise en charge ou choisissez manuellement ci-dessous.")
        # Fallback manuel : proposer la liste complète si besoin
        try:
            tous = conn.table("contrats_animaux").select(
                "proprietaire,animal,type_animal,date_naissance,identification,formule"
            ).limit(100).execute().data
        except Exception as e:
            tous = []
        if tous:
            choix = st.selectbox(
                "Sélectionnez un contrat manuellement :",
                [f"{r['proprietaire']} - {r['animal']} ({r['identification']})" for r in tous]
            )
            idx = [f"{r['proprietaire']} - {r['animal']} ({r['identification']})" for r in tous].index(choix)
            client = tous[idx]
        else:
            client = {
                "proprietaire": infos_client.get("nom_proprietaire") or "Simulation",
                "animal": infos_client.get("nom_animal") or "Simulation",
                "type_animal": "",
                "formule": select_formule()
            }

    # --- Affichage des infos client/animal ---
    st.sidebar.markdown(f"**Propriétaire :** {client.get('proprietaire','')}")
    st.sidebar.markdown(f"**Animal :** {client.get('animal','')} ({client.get('type_animal','')})")
    st.sidebar.markdown(f"**Formule :** {client.get('formule','')}")

    # --- Extraction complète et calcul du remboursement ---
    with st.spinner("Calcul du remboursement en cours..."):
        try:
            result = processor.process_facture_pennypet(
                file_bytes=bytes_data,
                formule_client=client["formule"],
                llm_provider=provider
            )
        except ValueError as ve:
            st.error(f"Erreur de parsing ou de validation JSON : {ve}")
            st.stop()
        except Exception as e:
            st.error(f"Erreur lors de l'analyse (réseau ou API) : {e}")
            st.stop()

    # --- Affichage du résultat de remboursement ---
    st.subheader("Détails du remboursement")
    st.json({
        "lignes":          result["remboursements"],
        "total_facture":   result["total_facture"],
        "total_remboursé": result["total_remboursement"],
        "reste_à_charge":  result["reste_total_a_charge"]
    })

    # --- Enregistrement optionnel (si contrat réel) ---
    if res and st.button("Enregistrer le remboursement"):
        try:
            data_id = (
                conn
                .table("contrats_animaux")
                .select("id")
                .eq("identification", client["identification"])
                .limit(1)
                .execute()
                .data
            )
            if not data_id:
                st.error("Aucun contrat trouvé pour cet identifiant.")
                st.stop()
            id_contrat = data_id[0]["id"]
            _ = (
                conn
                .table("remboursements")
                .insert([{
                    "id_contrat": id_contrat,
                    "date_acte": "now()",
                    "montant_facture": result["total_facture"],
                    "montant_rembourse": result["total_remboursement"],
                    "reste_a_charge": result["reste_total_a_charge"]
                }])
                .execute()
            )
            st.success("Opération enregistrée en base.")
        except Exception as e:
            st.error(f"Erreur lors de l'enregistrement en base : {e}")
else:
    st.info("Importez une facture pour démarrer l’analyse automatique et la simulation de remboursement.")
