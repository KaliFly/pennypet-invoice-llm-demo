# ui/app.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from st_supabase_connection import SupabaseConnection
from llm_parser.pennypet_processor import PennyPetProcessor

# Page config et style
st.set_page_config(
    page_title="PennyPet – Extraction & Remboursement",
    layout="wide"
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

# 2. Contrôles sidebar
with st.sidebar:
    st.header("Paramètres")
    provider = st.selectbox("Modèle Vision", ["qwen", "mistral"], index=0)
    formules_possibles = ["START", "PREMIUM", "INTEGRAL", "INTEGRAL_PLUS"]
    formule_simulation = st.selectbox("Formule (simulation)", formules_possibles, index=0)

# 3. Upload fichier
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
            formule_client="INTEGRAL",
            llm_provider=provider
        )
        if not isinstance(temp, dict):
            st.error("❌ Extraction échouée.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Erreur extraction : {e}")
        st.stop()

infos = temp.get("infos_client", {})
identification = infos.get("identification")
nom_proprietaire = infos.get("nom_proprietaire")
nom_animal = infos.get("nom_animal")

# 5. Vérification RPC pour Hélène Zambetti
terme = "%Hélène Zambetti%"
st.write("🔍 Vérification RPC avec terme :", terme)
resp = conn.rpc("search_contrat_by_name", {"term": terme}).execute()
st.write("▶︎ Statut RPC :", resp)
res_rpc = resp.data

# 6. Recherche dans la base
res = []
try:
    if identification:
        res = conn.table("contrats_animaux") \
            .select("proprietaire,animal,type_animal,date_naissance,identification,formule") \
            .eq("identification", identification) \
            .limit(1).execute().data
    elif nom_proprietaire:
        res = conn.rpc("search_contrat_by_name", {"term": terme}).execute().data
    elif nom_animal:
        res = conn.rpc("search_contrat_by_name", {"term": terme}).execute().data
except Exception as e:
    st.warning(f"⚠️ Recherche échouée : {e}")

# 7. Définir la formule
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
    st.warning("⚠️ Aucun contrat trouvé – mode simulation.")
    formule_client = formule_simulation
    client = {
        "proprietaire": nom_proprietaire or "Simulation",
        "animal": nom_animal or "Simulation",
        "type_animal": "",
        "formule": formule_client
    }

# 8. Affichage infos client
st.sidebar.markdown("### Client & Animal")
st.sidebar.markdown(f"**Propriétaire :** {client['proprietaire']}")
st.sidebar.markdown(f"**Animal :** {client['animal']} ({client['type_animal']})")
st.sidebar.markdown(f"**Formule :** {client['formule']}")

# 9. Calcul remboursement
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
        st.error(f"❌ Erreur : {e}")
        st.stop()

# 10. Affichage détail exhaustif
st.subheader("📊 Détails du remboursement")
import pandas as pd

df = pd.DataFrame(result["remboursements"])
df = df.rename(columns={
    "libelle_original":    "Libellé brut",
    "code_normalise":      "Code normalisé",
    "montant_ht":          "Montant HT (€)",
    "taux_applique":       "Taux appliqué (%)",
    "remboursement_brut":  "Remboursement brut (€)",
    "remboursement_final": "Remboursement final (€)",
    "reste_a_charge":      "Reste à charge (€)"
})
st.dataframe(df, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total facture", f"{result['total_facture']:.2f} €")
col2.metric("Total remboursé", f"{result['total_remboursement']:.2f} €")
col3.metric("Reste à charge", f"{result['reste_total_a_charge']:.2f} €")

# 11. Enregistrement
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
        st.error(f"❌ Erreur : {e}")
