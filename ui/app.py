import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
from datetime import datetime
from supabase import create_client
from llm_parser.pennypet_processor import PennyPetProcessor

# Configuration de la page
st.set_page_config(
    page_title="PennyPet – Extraction & Remboursement",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avancé
st.markdown("""
<style>
    /* Variables CSS */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --error-color: #F44336;
        --bg-color: #f8f9fa;
        --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Arrière-plan principal */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    /* Conteneur principal */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: var(--card-shadow);
    }

    /* Titre principal */
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Cartes d'information */
    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        box-shadow: var(--card-shadow);
        transition: transform 0.3s ease;
    }

    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    /* Métriques personnalisées */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        min-width: 150px;
        box-shadow: var(--card-shadow);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: scale(1.05);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Sidebar styling */
    .sidebar .stSelectbox > label {
        font-weight: 600;
        color: var(--primary-color);
    }

    .sidebar .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .sidebar .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed var(--primary-color);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .upload-zone:hover {
        border-color: var(--secondary-color);
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    }

    /* Alertes personnalisées */
    .alert {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }

    .alert-success {
        background: #d4edda;
        border-color: var(--success-color);
        color: #155724;
    }

    .alert-warning {
        background: #fff3cd;
        border-color: var(--warning-color);
        color: #856404;
    }

    .alert-error {
        background: #f8d7da;
        border-color: var(--error-color);
        color: #721c24;
    }

    .alert-info {
        background: #d1ecf1;
        border-color: var(--primary-color);
        color: #0c5460;
    }

    /* Spinners personnalisés */
    .stSpinner > div > div {
        border-top-color: var(--primary-color) !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: var(--card-shadow);
    }

    /* Progress bar */
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }

    .progress-bar {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        height: 20px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }

    /* Animation pour les cartes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Styles spécifiques pour les formules PennyPet */
    .formule-start {
        background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%);
        color: white;
    }

    .formule-premium {
        background: linear-gradient(135deg, #fd7e14 0%, #e8590c 100%);
        color: white;
    }

    .formule-integral {
        background: linear-gradient(135deg, #198754 0%, #146c43 100%);
        color: white;
    }

    .formule-integral-plus {
        background: linear-gradient(135deg, #6f42c1 0%, #59359a 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utilitaires
def display_alert(message, alert_type="info"):
    """Affiche une alerte stylisée"""
    st.markdown(f'<div class="alert alert-{alert_type}">{message}</div>', unsafe_allow_html=True)

def display_metric_card(value, label, color="primary"):
    """Affiche une métrique dans une carte stylisée"""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def validate_file(uploaded_file):
    """Valide le fichier uploadé"""
    if not uploaded_file:
        return False, "Aucun fichier sélectionné"
    
    if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
        return False, "Fichier trop volumineux (max 10MB)"
    
    if uploaded_file.type not in ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']:
        return False, "Type de fichier non supporté"
    
    return True, "Fichier valide"

def create_progress_bar(progress, total, label=""):
    """Crée une barre de progression personnalisée"""
    percentage = (progress / total) * 100 if total > 0 else 0
    return f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>{label}</span>
            <span>{percentage:.1f}%</span>
        </div>
        <div class="progress-bar" style="width: {percentage}%;"></div>
    </div>
    """

def get_pennypet_formule_info(formule):
    """Retourne les informations correctes des formules PennyPet"""
    formules_pennypet = {
        "START": {
            "type": "aucune_couverture",
            "taux_remboursement": 0,
            "plafond": 0,
            "description": "Pas d'assurance incluse",
            "couverture": "Aucune couverture",
            "color": "#6c757d"
        },
        "PREMIUM": {
            "type": "accident_uniquement", 
            "taux_remboursement": 100,
            "plafond": 500,
            "description": "Fonds d'urgence accident – prise en charge des frais consécutifs à un accident, jusqu'à 500€ par an",
            "couverture": "Accidents uniquement",
            "color": "#fd7e14"
        },
        "INTEGRAL": {
            "type": "accident_et_maladie",
            "taux_remboursement": 50,
            "plafond": 1000,
            "description": "Assurance santé animale – prise en charge à 50% des frais vétérinaires (accident & maladie), plafond 1000€ par an",
            "couverture": "Accidents et maladies",
            "color": "#198754"
        },
        "INTEGRAL_PLUS": {
            "type": "accident_et_maladie",
            "taux_remboursement": 100,
            "plafond": 1000,
            "description": "Assurance santé animale – prise en charge à 100% des frais vétérinaires (accident & maladie), plafond 1000€ par an",
            "couverture": "Accidents et maladies",
            "color": "#6f42c1"
        }
    }
    return formules_pennypet.get(formule, formules_pennypet["START"])

# Initialisation de la session
if 'processing_step' not in st.session_state:
    st.session_state.processing_step = 0
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'client_info' not in st.session_state:
    st.session_state.client_info = None

# En-tête principal
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🐕 PennyPet – Extraction & Remboursement</h1>', unsafe_allow_html=True)

# Connexion à Supabase
@st.cache_resource
def init_supabase():
    """Initialise la connexion Supabase"""
    if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
        st.error("❗ Veuillez définir SUPABASE_URL et SUPABASE_KEY dans vos secrets.")
        st.stop()
    
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# Sidebar avec paramètres
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    
    # Configuration du modèle
    with st.expander("🤖 Configuration IA", expanded=True):
        provider = st.selectbox(
            "Modèle Vision",
            ["qwen", "mistral"],
            index=0,
            help="Sélectionnez le modèle d'IA pour l'extraction"
        )
        
        formules_possibles = ["START", "PREMIUM", "INTEGRAL", "INTEGRAL_PLUS"]
        formule_simulation = st.selectbox(
            "Formule (simulation)",
            formules_possibles,
            index=0,
            help="Formule utilisée en mode simulation"
        )
    
    # Informations sur les formules PennyPet réelles
    with st.expander("📋 Formules PennyPet", expanded=True):
        for formule in formules_possibles:
            info = get_pennypet_formule_info(formule)
            st.markdown(f"""
            <div style="background: {info['color']}; color: white; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                <strong>{formule}</strong><br>
                {info['couverture']}<br>
                Remboursement: {info['taux_remboursement']}%<br>
                Plafond: {info['plafond']}€/an
            </div>
            """, unsafe_allow_html=True)
    
    # Statistiques de session
    if st.session_state.extraction_result:
        st.markdown("### 📊 Statistiques")
        stats = st.session_state.extraction_result.get('stats', {})
        st.metric("Lignes traitées", stats.get('lignes_traitees', 0))
        st.metric("Médicaments détectés", stats.get('medicaments_detectes', 0))
        st.metric("Actes détectés", stats.get('actes_detectes', 0))

# Interface principale avec tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload & Extraction", "🔍 Identification Client", "💰 Remboursement", "📈 Analyse"])

with tab1:
    st.markdown("### 📁 Import de Facture")
    
    # Zone d'upload améliorée
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Glissez-déposez votre facture ou cliquez pour sélectionner",
            type=["pdf", "jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded:
            # Validation du fichier
            is_valid, message = validate_file(uploaded)
            if is_valid:
                display_alert(f"✅ {message} - {uploaded.name} ({uploaded.size:,} bytes)", "success")
            else:
                display_alert(f"❌ {message}", "error")
                st.stop()
    
    with col2:
        if uploaded:
            # Prévisualisation pour les images
            if uploaded.type.startswith('image/'):
                try:
                    image = Image.open(uploaded)
                    st.image(image, caption="Prévisualisation", use_column_width=True)
                except Exception as e:
                    display_alert(f"Erreur prévisualisation: {e}", "warning")
            else:
                st.info("📄 Fichier PDF détecté\n\nLe fichier sera converti en image pour l'analyse.")
    
    # Traitement de l'extraction
    if uploaded and st.button("🚀 Analyser la facture", type="primary"):
        bytes_data = uploaded.read()
        if not bytes_data:
            display_alert("⚠️ Le fichier est vide ou corrompu.", "error")
            st.stop()
        
        processor = PennyPetProcessor()
        
        # Barre de progression
        progress_placeholder = st.empty()
        
        # Étape 1: Extraction
        progress_placeholder.markdown(create_progress_bar(1, 3, "🔍 Extraction des données..."), unsafe_allow_html=True)
        
        try:
            with st.spinner("Analyse en cours..."):
                temp = processor.process_facture_pennypet(
                    file_bytes=bytes_data,
                    formule_client="INTEGRAL",
                    llm_provider=provider
                )
                
                if not temp.get('success', False):
                    display_alert(f"❌ Extraction échouée: {temp.get('error', 'Erreur inconnue')}", "error")
                    st.stop()
                
                st.session_state.extraction_result = temp
                st.session_state.processing_step = 1
                
        except Exception as e:
            display_alert(f"❌ Erreur lors de l'extraction: {str(e)}", "error")
            st.stop()
        
        progress_placeholder.markdown(create_progress_bar(3, 3, "✅ Extraction terminée"), unsafe_allow_html=True)
        display_alert("🎉 Extraction réussie! Passez à l'onglet suivant.", "success")

with tab2:
    st.markdown("### 🔍 Identification du Client")
    
    if not st.session_state.extraction_result:
        display_alert("⚠️ Veuillez d'abord extraire les données de la facture dans l'onglet précédent.", "warning")
    else:
        # Récupération des informations client
        infos = st.session_state.extraction_result.get("informations_client", {})
        identification = infos.get("identification")
        nom_proprietaire = infos.get("nom_proprietaire")
        nom_animal = infos.get("nom_animal")
        
        # Affichage des informations extraites
        st.markdown("#### 📋 Informations extraites de la facture")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="info-card fade-in-up">
                <h4>👤 Propriétaire</h4>
                <p><strong>{nom_proprietaire or 'Non détecté'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card fade-in-up">
                <h4>🐾 Animal</h4>
                <p><strong>{nom_animal or 'Non détecté'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-card fade-in-up">
                <h4>🆔 Identification</h4>
                <p><strong>{identification or 'Non détecté'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recherche du contrat
        st.markdown("#### 🔗 Recherche du contrat")
        
        search_performed = False
        res = []
        
        if st.button("🔍 Rechercher le contrat", type="primary"):
            search_performed = True
            
            with st.spinner("Recherche en cours..."):
                try:
                    # Recherche par identification
                    if identification:
                        with st.expander("🔍 Recherche par identification", expanded=True):
                            st.info(f"Recherche pour: {identification}")
                            res = supabase.table("contrats_animaux") \
                                .select("proprietaire,animal,type_animal,date_naissance,identification,formule") \
                                .eq("identification", identification) \
                                .limit(1).execute().data
                            
                            if res:
                                st.success(f"✅ Contrat trouvé par identification!")
                            else:
                                st.warning("⚠️ Aucun contrat trouvé par identification")
                    
                    # Fallback par nom de propriétaire
                    if not res and nom_proprietaire:
                        with st.expander("🔍 Recherche par nom de propriétaire", expanded=True):
                            terme = f"%{nom_proprietaire.strip()}%"
                            st.info(f"Recherche pour: {terme}")
                            rpc_resp = supabase.rpc("search_contrat_by_name", {"term": terme}).execute()
                            res = rpc_resp.data
                            
                            if res:
                                st.success(f"✅ {len(res)} contrat(s) trouvé(s) par nom!")
                            else:
                                st.warning("⚠️ Aucun contrat trouvé par nom de propriétaire")
                    
                    # Fallback par nom d'animal
                    if not res and nom_animal:
                        with st.expander("🔍 Recherche par nom d'animal", expanded=True):
                            terme = f"%{nom_animal.strip()}%"
                            st.info(f"Recherche pour: {terme}")
                            rpc_resp = supabase.rpc("search_contrat_by_name", {"term": terme}).execute()
                            res = rpc_resp.data
                            
                            if res:
                                st.success(f"✅ {len(res)} contrat(s) trouvé(s) par nom d'animal!")
                            else:
                                st.warning("⚠️ Aucun contrat trouvé par nom d'animal")
                                
                except Exception as e:
                    display_alert(f"❌ Erreur lors de la recherche: {str(e)}", "error")
        
        # Gestion des résultats de recherche
        if search_performed:
            if res and len(res) == 1:
                client = res[0]
                formule_client = client["formule"]
                st.session_state.client_info = client
                
                st.markdown("#### ✅ Contrat identifié")
                
                # Affichage avec les vraies informations PennyPet
                info_formule = get_pennypet_formule_info(formule_client)
                st.markdown(f"""
                <div class="info-card fade-in-up">
                    <h4>📋 Informations du contrat</h4>
                    <p><strong>Propriétaire:</strong> {client['proprietaire']}</p>
                    <p><strong>Animal:</strong> {client['animal']} ({client.get('type_animal', 'Non spécifié')})</p>
                    <p><strong>Identification:</strong> {client['identification']}</p>
                    <div style="background: {info_formule['color']}; color: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <h5>📊 Formule {client['formule']}</h5>
                        <p><strong>Couverture:</strong> {info_formule['couverture']}</p>
                        <p><strong>Remboursement:</strong> {info_formule['taux_remboursement']}%</p>
                        <p><strong>Plafond:</strong> {info_formule['plafond']}€/an</p>
                        <p><em>{info_formule['description']}</em></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                display_alert("🎉 Contrat trouvé! Passez à l'onglet Remboursement.", "success")
                
            elif res and len(res) > 1:
                st.markdown("#### 🔄 Plusieurs contrats trouvés")
                
                # Affichage des options
                options = [f"{r['proprietaire']} – {r['animal']} ({r['identification']})" for r in res]
                choix = st.selectbox("Sélectionnez le contrat approprié:", options)
                
                if choix:
                    idx = options.index(choix)
                    client = res[idx]
                    formule_client = client["formule"]
                    st.session_state.client_info = client
                    
                    # Affichage avec les vraies informations PennyPet
                    info_formule = get_pennypet_formule_info(formule_client)
                    st.markdown(f"""
                    <div class="info-card fade-in-up">
                        <h4>📋 Contrat sélectionné</h4>
                        <p><strong>Propriétaire:</strong> {client['proprietaire']}</p>
                        <p><strong>Animal:</strong> {client['animal']} ({client.get('type_animal', 'Non spécifié')})</p>
                        <div style="background: {info_formule['color']}; color: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                            <h5>📊 Formule {client['formule']}</h5>
                            <p><strong>Couverture:</strong> {info_formule['couverture']}</p>
                            <p><strong>Remboursement:</strong> {info_formule['taux_remboursement']}%</p>
                            <p><strong>Plafond:</strong> {info_formule['plafond']}€/an</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    display_alert("✅ Contrat sélectionné! Passez à l'onglet Remboursement.", "success")
            else:
                st.markdown("#### ⚠️ Mode Simulation")
                display_alert("Aucun contrat trouvé dans la base de données. Utilisation du mode simulation.", "warning")
                
                # Formulaire de simulation
                with st.form("simulation_form"):
                    st.markdown("##### 📝 Informations pour la simulation")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        sim_proprietaire = st.text_input("Nom du propriétaire", value=nom_proprietaire or "")
                        sim_animal = st.text_input("Nom de l'animal", value=nom_animal or "")
                    
                    with col2:
                        sim_type = st.selectbox("Type d'animal", ["Chien", "Chat", "NAC", "Autre"])
                        sim_formule = st.selectbox("Formule à simuler", formules_possibles)
                    
                    if st.form_submit_button("💻 Utiliser la simulation"):
                        client = {
                            "proprietaire": sim_proprietaire or "Simulation",
                            "animal": sim_animal or "Simulation",
                            "type_animal": sim_type,
                            "formule": sim_formule,
                            "identification": "SIMULATION"
                        }
                        st.session_state.client_info = client
                        
                        st.success("✅ Mode simulation activé! Passez à l'onglet Remboursement.")

with tab3:
    st.markdown("### 💰 Calcul du Remboursement")
    
    if not st.session_state.extraction_result or not st.session_state.client_info:
        display_alert("⚠️ Veuillez d'abord extraire les données et identifier le client.", "warning")
    else:
        client = st.session_state.client_info
        formule_client = client["formule"]
        
        # Affichage des informations client
        st.markdown("#### 👤 Informations du client")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <h4>📋 Détails du contrat</h4>
                <p><strong>Propriétaire:</strong> {client['proprietaire']}</p>
                <p><strong>Animal:</strong> {client['animal']} ({client.get('type_animal', 'Non spécifié')})</p>
                <p><strong>Formule:</strong> <span style="color: var(--primary-color); font-weight: bold;">{client['formule']}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Informations sur la formule PennyPet réelle
            info_formule = get_pennypet_formule_info(formule_client)
            st.markdown(f"""
            <div class="info-card">
                <div style="background: {info_formule['color']}; color: white; padding: 1rem; border-radius: 8px;">
                    <h4>📊 Couverture {formule_client}</h4>
                    <p><strong>Type:</strong> {info_formule['couverture']}</p>
                    <p><strong>Remboursement:</strong> {info_formule['taux_remboursement']}%</p>
                    <p><strong>Plafond annuel:</strong> {info_formule['plafond']}€</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Calcul du remboursement
        if st.button("💳 Calculer le remboursement", type="primary"):
            processor = PennyPetProcessor()
            
            with st.spinner("Calcul du remboursement en cours..."):
                try:
                    # Récupération des bytes de la facture
                    bytes_data = uploaded.read() if uploaded else None
                    if not bytes_data:
                        display_alert("❌ Données de facture non disponibles", "error")
                        st.stop()
                    
                    result = processor.process_facture_pennypet(
                        file_bytes=bytes_data,
                        formule_client=formule_client,
                        llm_provider=provider
                    )
                    
                    if not result.get('success', False):
                        display_alert(f"❌ Calcul échoué: {result.get('error', 'Erreur inconnue')}", "error")
                        st.stop()
                    
                    # Stockage du résultat
                    st.session_state.remboursement_result = result
                    
                    # Affichage des résultats
                    st.markdown("#### 📊 Résultats du remboursement")
                    
                    # Calcul des totaux selon les vraies règles PennyPet
                    total_facture = 0
                    total_rembourse = 0
                    
                    if result.get('lignes'):
                        for item in result['lignes']:
                            ligne = item.get('ligne', {})
                            montant_ht = ligne.get('montant_ht', 0)
                            total_facture += montant_ht
                            
                            # Application des vraies règles PennyPet
                            if formule_client == "START":
                                # Aucune couverture
                                total_rembourse += 0
                            elif formule_client == "PREMIUM":
                                # Accidents uniquement, 100% jusqu'à 500€
                                est_accident = item.get('est_accident', False)
                                if est_accident:
                                    remb = min(montant_ht, 500)  # Plafond 500€
                                    total_rembourse += remb
                            elif formule_client == "INTEGRAL":
                                # Accidents et maladies, 50% jusqu'à 1000€
                                remb = min(montant_ht * 0.5, 1000)  # 50% avec plafond 1000€
                                total_rembourse += remb
                            elif formule_client == "INTEGRAL_PLUS":
                                # Accidents et maladies, 100% jusqu'à 1000€
                                remb = min(montant_ht, 1000)  # 100% avec plafond 1000€
                                total_rembourse += remb
                    
                    reste_a_charge = total_facture - total_rembourse
                    
                    # Affichage des métriques
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(display_metric_card(f"{total_facture:.2f}€", "Total Facture"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(display_metric_card(f"{total_rembourse:.2f}€", "Total Remboursé"), unsafe_allow_html=True)
                    with col3:
                        st.markdown(display_metric_card(f"{reste_a_charge:.2f}€", "Reste à Charge"), unsafe_allow_html=True)
                    
                    # Tableau détaillé
                    st.markdown("#### 📋 Détail par ligne")
                    
                    if result.get('lignes'):
                        # Préparation des données pour le tableau
                        lignes_data = []
                        for item in result['lignes']:
                            ligne = item.get('ligne', {})
                            montant_ht = ligne.get('montant_ht', 0)
                            
                            # Calcul du remboursement selon la formule
                            remb_individuel = 0
                            if formule_client == "START":
                                remb_individuel = 0
                            elif formule_client == "PREMIUM":
                                est_accident = item.get('est_accident', False)
                                if est_accident:
                                    remb_individuel = min(montant_ht, 500)
                            elif formule_client == "INTEGRAL":
                                remb_individuel = min(montant_ht * 0.5, 1000)
                            elif formule_client == "INTEGRAL_PLUS":
                                remb_individuel = min(montant_ht, 1000)
                            
                            reste_individuel = montant_ht - remb_individuel
                            
                            lignes_data.append({
                                'Description': ligne.get('description', ligne.get('code_acte', 'Non spécifié')),
                                'Type': '💊 Médicament' if ligne.get('est_medicament') else '🏥 Acte',
                                'Montant HT': f"{montant_ht:.2f}€",
                                'Formule': formule_client,
                                'Remboursé': f"{remb_individuel:.2f}€",
                                'Reste': f"{reste_individuel:.2f}€"
                            })
                        
                        df = pd.DataFrame(lignes_data)
                        st.dataframe(df, use_container_width=True)
                    
                    # Explication des règles appliquées
                    st.markdown("#### 📖 Règles appliquées")
                    info_formule = get_pennypet_formule_info(formule_client)
                    
                    if formule_client == "START":
                        st.info("🔍 **Formule START**: Aucune couverture d'assurance incluse")
                    elif formule_client == "PREMIUM":
                        st.info("🚨 **Formule PREMIUM**: Fonds d'urgence pour accidents uniquement - Remboursement à 100% jusqu'à 500€/an")
                    elif formule_client == "INTEGRAL":
                        st.info("💚 **Formule INTEGRAL**: Assurance santé animale - Remboursement à 50% pour accidents et maladies jusqu'à 1000€/an")
                    elif formule_client == "INTEGRAL_PLUS":
                        st.info("💜 **Formule INTEGRAL_PLUS**: Assurance santé animale - Remboursement à 100% pour accidents et maladies jusqu'à 1000€/an")
                    
                    # Bouton d'enregistrement
                    if client.get('identification') != 'SIMULATION':
                        if st.button("💾 Enregistrer le remboursement", type="secondary"):
                            try:
                                # Recherche de l'ID du contrat
                                contrat_data = supabase.table("contrats_animaux") \
                                    .select("id") \
                                    .eq("identification", client["identification"]) \
                                    .limit(1).execute().data
                                
                                if contrat_data:
                                    contrat_id = contrat_data[0]["id"]
                                    
                                    # Insertion du remboursement
                                    supabase.table("remboursements").insert([{
                                        "id_contrat": contrat_id,
                                        "date_acte": datetime.now().isoformat(),
                                        "montant_facture": total_facture,
                                        "montant_rembourse": total_rembourse,
                                        "reste_a_charge": reste_a_charge
                                    }]).execute()
                                    
                                    display_alert("✅ Remboursement enregistré avec succès!", "success")
                                else:
                                    display_alert("❌ Impossible de trouver le contrat", "error")
                                    
                            except Exception as e:
                                display_alert(f"❌ Erreur lors de l'enregistrement: {str(e)}", "error")
                    
                except Exception as e:
                    display_alert(f"❌ Erreur lors du calcul: {str(e)}", "error")

with tab4:
    st.markdown("### 📈 Analyse et Statistiques")
    
    if not st.session_state.get('remboursement_result'):
        display_alert("⚠️ Veuillez d'abord calculer le remboursement pour voir l'analyse.", "warning")
    else:
        result = st.session_state.remboursement_result
        client = st.session_state.client_info
        
        # Graphique de répartition
        st.markdown("#### 📊 Répartition des coûts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique en camembert
            total_facture = sum(item.get('ligne', {}).get('montant_ht', 0) for item in result.get('lignes', []))
            total_rembourse = 0
            
            # Calcul selon les vraies règles PennyPet
            formule_client = client["formule"]
            for item in result.get('lignes', []):
                ligne = item.get('ligne', {})
                montant_ht = ligne.get('montant_ht', 0)
                
                if formule_client == "START":
                    total_rembourse += 0
                elif formule_client == "PREMIUM":
                    est_accident = item.get('est_accident', False)
                    if est_accident:
                        total_rembourse += min(montant_ht, 500)
                elif formule_client == "INTEGRAL":
                    total_rembourse += min(montant_ht * 0.5, 1000)
                elif formule_client == "INTEGRAL_PLUS":
                    total_rembourse += min(montant_ht, 1000)
            
            reste_a_charge = total_facture - total_rembourse
            
            if total_rembourse > 0 or reste_a_charge > 0:
                fig_pie = px.pie(
                    values=[total_rembourse, reste_a_charge],
                    names=['Remboursé', 'Reste à charge'],
                    title="Répartition des coûts selon PennyPet",
                    color_discrete_sequence=['#4CAF50', '#FF5722']
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Graphique par type (médicaments vs actes)
            if result.get('lignes'):
                medicaments_total = sum(
                    item.get('ligne', {}).get('montant_ht', 0) 
                    for item in result['lignes'] 
                    if item.get('ligne', {}).get('est_medicament', False)
                )
                actes_total = sum(
                    item.get('ligne', {}).get('montant_ht', 0) 
                    for item in result['lignes'] 
                    if not item.get('ligne', {}).get('est_medicament', False)
                )
                
                if medicaments_total > 0 or actes_total > 0:
                    fig_bar = px.bar(
                        x=['Médicaments', 'Actes'],
                        y=[medicaments_total, actes_total],
                        title="Répartition par type",
                        color=['Médicaments', 'Actes'],
                        color_discrete_sequence=['#2196F3', '#FF9800']
                    )
                    fig_bar.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        # Statistiques détaillées
        st.markdown("#### 📋 Statistiques détaillées")
        
        stats = result.get('statistiques', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lignes traitées", stats.get('lignes_traitees', 0))
        with col2:
            st.metric("Médicaments détectés", stats.get('medicaments_detectes', 0))
        with col3:
            st.metric("Actes détectés", stats.get('actes_detectes', 0))
        with col4:
            taux_global = (total_rembourse / total_facture * 100) if total_facture > 0 else 0
            st.metric("Taux global", f"{taux_global:.1f}%")
        
        # Tableau de comparaison des formules PennyPet réelles
        st.markdown("#### 💡 Comparaison des formules PennyPet")
        
        formules_comp = {
            "Formule": ["START", "PREMIUM", "INTEGRAL", "INTEGRAL_PLUS"],
            "Couverture": ["Aucune", "Accidents", "Accidents + Maladies", "Accidents + Maladies"],
            "Taux": ["0%", "100%", "50%", "100%"],
            "Plafond": ["0€", "500€", "1000€", "1000€"],
            "Remboursement estimé": []
        }
        
        # Calcul estimé pour chaque formule selon les vraies règles
        for formule in formules_comp["Formule"]:
            if formule == "START":
                estim = 0
            elif formule == "PREMIUM":
                # Seulement les accidents
                accidents_total = sum(
                    item.get('ligne', {}).get('montant_ht', 0) 
                    for item in result.get('lignes', [])
                    if item.get('est_accident', False)
                )
                estim = min(accidents_total, 500)
            elif formule == "INTEGRAL":
                estim = min(total_facture * 0.5, 1000)
            elif formule == "INTEGRAL_PLUS":
                estim = min(total_facture, 1000)
            
            formules_comp["Remboursement estimé"].append(f"{estim:.2f}€")
        
        df_comp = pd.DataFrame(formules_comp)
        st.dataframe(df_comp, use_container_width=True)
        
        # Délais de carence PennyPet
        st.markdown("#### ⏰ Délais de carence PennyPet")
        
        st.info("""
        **Délais de carence selon les conditions PennyPet:**[3][5][7]
        - 🚨 **Accidents**: 3 jours après souscription
        - 🏥 **Maladies**: 45 jours avant activation de la couverture  
        - 🔧 **Interventions chirurgicales** (suite à maladie): 120 jours de délai de carence
        """)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🐕 <strong>PennyPet</strong> - Votre assistant intelligent pour les remboursements vétérinaires</p>
    <p>Développé avec les vraies règles de prise en charge PennyPet</p>
</div>
""", unsafe_allow_html=True)
