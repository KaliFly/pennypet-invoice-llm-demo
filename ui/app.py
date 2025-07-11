# ui/app.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from st_supabase_connection import SupabaseConnection
from llm_parser.pennypet_processor import PennyPetProcessor

# Page configuration and styling
st.set_page_config(
    page_title="PennyPet Invoice + DB",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .stApp { background-color: #f9f9f9; }
    .title { color: #2E86AB; font-size:32px; font-weight:600; }
    .sidebar .stButton>button { background-color: #2E86AB; color: white; }
    .stSpinner>div>div { border-top-color: #2E86AB !important; }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<h1 class="title">PennyPet – Extraction & Remboursement</h1>', unsafe_allow_html=True)

# 1. Connexion à Supabase
if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error("❗ Veuillez définir SUPABASE_URL et SUPABASE_KEY dans vos secrets.")
    st.stop()
try:
    conn = st.connection(
        "supabase",
        type=SupabaseConnection,
        url=st.secrets["SUPABASE_URL"],
        key=st.secrets["SUPABASE_KEY"]
    )
except Exception as e:
    st.error(f"❌ Erreur de connexion à Supabase : {e}")
    st.stop()

# 2. Sidebar controls
with st.sidebar:
    st.header("Paramètres")
    provider = st.selectbox("Modèle Vision", ["qwen", "mistral"], index=0)
    formules_possibles = ["START", "PREMIUM", "INTEGRAL", "INTEGRAL_PLUS"]
    formule_simulation = st.selectbox("Formule (simulation)", formules_possibles, index=0)

# 3. File uploader
st.subheader("Importez votre facture")
uploaded = st.file_uploader("", type=["pdf", "jpg", "png"], label_visibility="collapsed")
if not uploaded:
    st.info("📄 Déposez un PDF, JPG ou PNG pour commencer.")
    st.stop()

bytes_data = uploaded.read()
if not bytes_data:
    st.error("⚠️ Le fichier est vide ou corrompu.")
    st.stop()

processor = PennyPetProcessor()

# 4. Extraction initiale
with st.spinner("🔍 Extraction des infos client..."):
    try:
        temp = processor.process_facture_pennypet(
            file_bytes=bytes_data,
            formule_client="INTEGRAL",  # neutre pour extraction
            llm_provider=provider
        )
        if not isinstance(temp, dict):
            st.error("❌ Extraction échouée.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Erreur extraction : {e}")
        st.stop()

infos = temp.get("infos_client", {})
identification   = infos.get("identification")
nom_proprietaire = infos.get("nom_proprietaire")
nom_animal       = infos.get("nom_animal")

# 5. Recherche du contrat
res = []
with st.spinner("🔗 Recherche du contrat…"):
    try:
        if identification:
            res = conn.table("contrats_animaux") \
                .select("proprietaire,animal,type_animal,date_naissance,identification,formule") \
                .eq("identification", identification) \
                .limit(1).execute().data
        elif nom_proprietaire:
            terme = f"%{nom_proprietaire.strip()}%"
            res = conn.rpc("search_contrat_by_name", {"term": terme}).execute().data
        elif nom_animal:
            terme = f"%{nom_animal.strip()}%"
            res = conn.rpc("search_contrat_by_name", {"term": terme}).execute().data
    except Exception as e:
        st.warning(f"⚠️ Recherche échouée : {e}")

# 6. Détermination de la formule cliente
if res and len(res) == 1:
    client = res[0]
    formule_client = client["formule"]
elif res and len(res) > 1:
    choix = st.selectbox(
        "Plusieurs contrats trouvés",
        [f"{r['proprietaire']} – {r['animal']} ({r['identification']})" for r in res]
    )
    idx = [f"{r['proprietaire']} – {r['animal']} ({r['identification']})" for r in res].index(choix)
    client = res[idx]
    formule_client = client["formule"]
else:
    st.warning("⚠️ Aucun contrat trouvé – mode simulation activé.")
    formule_client = formule_simulation
    client = {
        "proprietaire": nom_proprietaire or "Simulation",
        "animal":       nom_animal or "Simulation",
        "type_animal":  "",
        "formule":      formule_client
    }

# 7. Affichage infos client/animal
st.sidebar.markdown("### Client & Animal")
st.sidebar.markdown(f"**Propriétaire :** {client['proprietaire']}")
st.sidebar.markdown(f"**Animal :** {client['animal']} ({client['type_animal']})")
st.sidebar.markdown(f"**Formule :** {client['formule']}")

# 8. Calcul du remboursement
with st.spinner("⏳ Calcul du remboursement..."):
    try:
        result = processor.process_facture_pennypet(
            file_bytes=bytes_data,
            formule_client=formule_client,
            llm_provider=provider
        )
        if not isinstance(result, dict):
            st.error("❌ Calcul échoué.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Erreur calcul : {e}")
        st.stop()

# 9. Affichage du résultat
st.subheader("📊 Détails du remboursement")
try:
    st.json({
        "lignes":          result["remboursements"],
        "total_facture":   result["total_facture"],
        "total_remboursé": result["total_remboursement"],
        "reste_à_charge":  result["reste_total_a_charge"]
    }, expanded=False)
except Exception as e:
    st.error(f"❌ Erreur affichage : {e}")

# 10. Enregistrement optionnel
if res and st.button("💾 Enregistrer le remboursement"):
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
        st.success("✅ Remboursement enregistré.")
    except Exception as e:
        st.error(f"❌ Erreur enregistrement : {e}")
