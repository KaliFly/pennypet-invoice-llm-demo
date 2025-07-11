# ui/app.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from st_supabase_connection import SupabaseConnection
from llm_parser.pennypet_processor import PennyPetProcessor

st.set_page_config(page_title="PennyPet Invoice + DB", layout="wide")
st.title("PennyPet – Extraction & Remboursement")

# 1. Connexion à Supabase
if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error("Veuillez définir SUPABASE_URL et SUPABASE_KEY dans vos secrets.")
    st.stop()
try:
    conn = st.connection(
        "supabase",
        type=SupabaseConnection,
        url=st.secrets["SUPABASE_URL"],
        key=st.secrets["SUPABASE_KEY"]
    )
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
if not uploaded:
    st.info("Importez une facture pour démarrer l’analyse.")
    st.stop()

bytes_data = uploaded.read()
if not bytes_data:
    st.error("Le fichier uploadé est vide ou corrompu.")
    st.stop()

processor = PennyPetProcessor()

# 4. Extraction initiale pour récupérer infos_client
with st.spinner("Extraction des informations client..."):
    try:
        temp = processor.process_facture_pennypet(
            file_bytes=bytes_data,
            formule_client="INTEGRAL",  # formule neutre pour extraction
            llm_provider=provider
        )
        if not isinstance(temp, dict):
            st.error("Impossible d'extraire les informations de la facture.")
            st.stop()
    except Exception as e:
        st.error(f"Erreur extraction : {e}")
        st.stop()

infos = temp.get("infos_client", {})
identification   = infos.get("identification")
nom_proprietaire = infos.get("nom_proprietaire")
nom_animal       = infos.get("nom_animal")

# 5. Recherche du contrat (RPC accent/ordre-insensible pour proprietaire)
res = []
try:
    if identification:
        res = conn.table("contrats_animaux") \
            .select("proprietaire,animal,type_animal,date_naissance,identification,formule") \
            .eq("identification", identification) \
            .limit(1) \
            .execute().data
    elif nom_proprietaire:
        terme = f"%{nom_proprietaire.strip()}%"
        res = conn.rpc("search_contrat_by_name", {"term": terme}).execute().data
    elif nom_animal:
        terme = f"%{nom_animal.strip()}%"
        res = conn.rpc("search_contrat_by_name", {"term": terme}).execute().data
except Exception as e:
    st.warning(f"Recherche contrat échouée : {e}")

# 6. Détermination de la formule_client
if res and len(res) == 1:
    client = res[0]
    formule_client = client["formule"]
elif res and len(res) > 1:
    choix = st.selectbox(
        "Plusieurs contrats trouvés, sélectionnez :",
        [f"{r['proprietaire']} – {r['animal']} ({r['identification']})" for r in res]
    )
    idx = [f"{r['proprietaire']} – {r['animal']} ({r['identification']})" for r in res].index(choix)
    client = res[idx]
    formule_client = client["formule"]
else:
    st.warning("Aucun contrat trouvé. Simulation manuelle.")
    formule_client = select_formule()
    client = {
        "proprietaire": nom_proprietaire or "Simulation",
        "animal":       nom_animal or "Simulation",
        "type_animal":  "",
        "formule":      formule_client
    }

# 7. Affichage infos client
st.sidebar.markdown(f"**Propriétaire :** {client['proprietaire']}")
st.sidebar.markdown(f"**Animal :** {client['animal']} ({client['type_animal']})")
st.sidebar.markdown(f"**Formule :** {client['formule']}")

# 8. Traitement complet avec formule_client
with st.spinner("Analyse et calcul du remboursement..."):
    try:
        result = processor.process_facture_pennypet(
            file_bytes=bytes_data,
            formule_client=formule_client,
            llm_provider=provider
        )
        if not isinstance(result, dict):
            st.error("Le traitement n’a retourné aucun résultat exploitable.")
            st.stop()
    except Exception as e:
        st.error(f"Erreur calcul remboursement : {e}")
        st.stop()

# 9. Affichage du détail du remboursement
st.subheader("Détails du remboursement")
try:
    st.json({
        "lignes":         result["remboursements"],
        "total_facture":  result["total_facture"],
        "total_remboursé": result["total_remboursement"],
        "reste_à_charge": result["reste_total_a_charge"]
    })
except Exception as e:
    st.error(f"Erreur affichage résultat : {e}")

# 10. Enregistrement en base (optionnel)
if res and st.button("Enregistrer le remboursement"):
    try:
        contrat_id = conn.table("contrats_animaux") \
            .select("id") \
            .eq("identification", client["identification"]) \
            .limit(1).execute().data[0]["id"]
        conn.table("remboursements").insert([{
            "id_contrat":       contrat_id,
            "date_acte":        "now()",
            "montant_facture":  result["total_facture"],
            "montant_rembourse": result["total_remboursement"],
            "reste_a_charge":   result["reste_total_a_charge"]
        }]).execute()
        st.success("Remboursement enregistré.")
    except Exception as e:
        st.error(f"Erreur enregistrement : {e}")
