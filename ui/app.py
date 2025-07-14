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
    page_title="🐾 PennyPet – Remboursements Vétérinaires",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec fond sobre et gradients PennyPet sur le titre
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Palette PennyPet avec gradients rose-violet */
    :root {
        --pennypet-primary: #FF6B35;      /* Orange signature */
        --pennypet-secondary: #4ECDC4;    /* Turquoise bien-être */
        --pennypet-accent: #45B7D1;       /* Bleu océan Biarritz */
        --pennypet-success: #96CEB4;      /* Vert solidaire */
        --pennypet-warning: #FFEAA7;      /* Jaune attention */
        --pennypet-purple: #A29BFE;       /* Violet premium */
        --pennypet-pink: #fd79a8;         /* Rose signature */
        --pennypet-dark: #2D3436;         /* Gris foncé */
        --pennypet-light: #F8F9FA;        /* Blanc cassé */
        --pennypet-gradient: linear-gradient(135deg, #FF6B35 0%, #4ECDC4 100%);
        --pennypet-gradient-title: linear-gradient(135deg, #fd79a8 0%, #A29BFE 50%, #6c5ce7 100%);
        --pennypet-gradient-reverse: linear-gradient(135deg, #4ECDC4 0%, #FF6B35 100%);
        --pennypet-shadow: 0 8px 32px rgba(255, 107, 53, 0.1);
    }

    /* Font family */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* NOUVEAU : Arrière-plan sobre et élégant */
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%);
        min-height: 100vh;
    }

    /* Conteneur principal sobre */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(10px);
    }

    /* TITRE PRINCIPAL avec gradients rose-violet PennyPet */
    .pennypet-header {
        text-align: center;
        background: var(--pennypet-gradient-title);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
        text-shadow: 0 2px 4px rgba(253, 121, 168, 0.1);
    }

    /* Sous-titre avec couleur douce */
    .pennypet-subtitle {
        text-align: center;
        color: var(--pennypet-dark);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
        line-height: 1.6;
    }

    /* Cartes PennyPet avec fond sobre */
    .pennypet-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--pennypet-primary);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }

    .pennypet-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        border-left-color: var(--pennypet-secondary);
    }

    /* Métriques avec gradients PennyPet */
    .pennypet-metric {
        background: var(--pennypet-gradient);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.2);
        transition: transform 0.3s ease;
        margin: 0.5rem;
    }

    .pennypet-metric:hover {
        transform: scale(1.05);
    }

    .pennypet-metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .pennypet-metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }

    /* Boutons PennyPet avec gradients conservés */
    .stButton > button {
        background: var(--pennypet-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4) !important;
    }

    /* Alertes PennyPet avec fond sobre */
    .pennypet-alert-success {
        background: linear-gradient(135deg, rgba(150, 206, 180, 0.1) 0%, rgba(255, 234, 167, 0.1) 100%);
        border: 1px solid rgba(150, 206, 180, 0.3);
        color: var(--pennypet-dark);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--pennypet-success);
        font-weight: 500;
    }

    .pennypet-alert-info {
        background: linear-gradient(135deg, rgba(69, 183, 209, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
        border: 1px solid rgba(69, 183, 209, 0.3);
        color: var(--pennypet-dark);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--pennypet-accent);
        font-weight: 500;
    }

    .pennypet-alert-warning {
        background: linear-gradient(135deg, rgba(255, 234, 167, 0.1) 0%, rgba(255, 107, 53, 0.1) 100%);
        border: 1px solid rgba(255, 107, 53, 0.3);
        color: var(--pennypet-dark);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--pennypet-warning);
        font-weight: 500;
    }

    .pennypet-alert-error {
        background: linear-gradient(135deg, rgba(253, 121, 168, 0.1) 0%, rgba(232, 67, 147, 0.1) 100%);
        border: 1px solid rgba(253, 121, 168, 0.3);
        color: var(--pennypet-dark);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--pennypet-pink);
        font-weight: 500;
    }

    /* Sidebar conserve ses gradients PennyPet */
    .css-1d391kg {
        background: linear-gradient(180deg, #fd79a8 0%, #A29BFE 50%, #6c5ce7 100%);
    }

    .sidebar .stSelectbox > label {
        color: white !important;
        font-weight: 600 !important;
    }

    /* Upload zone sobre */
    .pennypet-upload {
        border: 2px dashed var(--pennypet-primary);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(255, 245, 243, 0.5) 0%, rgba(240, 253, 252, 0.5) 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .pennypet-upload:hover {
        border-color: var(--pennypet-secondary);
        background: linear-gradient(135deg, rgba(240, 253, 252, 0.5) 0%, rgba(255, 245, 243, 0.5) 100%);
        transform: translateY(-2px);
    }

    /* Animation PennyPet conservée */
    @keyframes pennypet-fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .pennypet-animated {
        animation: pennypet-fadeIn 0.6s ease-out;
    }

    /* Tabs avec fond sobre */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(248, 249, 250, 0.8);
        padding: 0.5rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: var(--pennypet-dark);
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    .stTabs [aria-selected="true"] {
        background: var(--pennypet-gradient-title);
        color: white;
        box-shadow: 0 4px 15px rgba(253, 121, 168, 0.3);
    }

    /* Progress bar sobre */
    .pennypet-progress {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }

    .pennypet-progress-bar {
        background: var(--pennypet-gradient-title);
        height: 20px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }

    /* Formule cards avec couleurs conservées */
    .formule-start { border-left-color: #6c757d !important; }
    .formule-premium { border-left-color: var(--pennypet-primary) !important; }
    .formule-integral { border-left-color: var(--pennypet-secondary) !important; }
    .formule-integral-plus { border-left-color: var(--pennypet-purple) !important; }

    /* Metrics grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }

    /* Dataframe styling sobre */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Messages PennyPet
PENNYPET_MESSAGES = {
    "welcome": "🐾 Bienvenue sur PennyPet ! Ensemble, simplifions la gestion des soins de ton animal",
    "upload_success": "✅ Super ! Ta facture a été analysée avec succès. Découvrons ensemble ce que PennyPet peut rembourser",
    "search_client": "🔍 Recherchons ton contrat PennyPet pour personnaliser tes remboursements",
    "no_contract": "💡 Pas de souci ! Utilisons le mode simulation pour découvrir ce que PennyPet pourrait t'apporter",
    "calculation_done": "🎉 Parfait ! Voici le détail de tes remboursements PennyPet",
    "save_success": "💾 Excellent ! Ton remboursement a été enregistré. L'équipe PennyPet s'occupe du reste !",
    "error": "😔 Oups ! Une petite erreur s'est glissée. L'équipe PennyPet va régler ça rapidement"
}

# Fonctions utilitaires PennyPet
def display_pennypet_alert(message, alert_type="info", emoji="✨"):
    """Affiche une alerte dans le style PennyPet"""
    st.markdown(f"""
    <div class="pennypet-alert-{alert_type} pennypet-animated">
        {emoji} {message}
    </div>
    """, unsafe_allow_html=True)

def display_pennypet_metric(value, label, emoji="📊"):
    """Affiche une métrique dans le style PennyPet"""
    return f"""
    <div class="pennypet-metric pennypet-animated">
        <div class="pennypet-metric-value">{emoji} {value}</div>
        <div class="pennypet-metric-label">{label}</div>
    </div>
    """

def create_pennypet_progress(progress, total, label=""):
    """Crée une barre de progression PennyPet"""
    percentage = (progress / total) * 100 if total > 0 else 0
    return f"""
    <div class="pennypet-progress">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; color: var(--pennypet-dark); font-weight: 500;">
            <span>🚀 {label}</span>
            <span>{percentage:.1f}%</span>
        </div>
        <div class="pennypet-progress-bar" style="width: {percentage}%;"></div>
    </div>
    """

def get_pennypet_formule_info(formule):
    """Retourne les informations PennyPet des formules"""
    formules_pennypet = {
        "START": {
            "type": "aucune_couverture",
            "taux_remboursement": 0,
            "plafond": 0,
            "description": "Pas d'assurance incluse",
            "couverture": "Aucune couverture",
            "color": "#6c757d",
            "emoji": "📱"
        },
        "PREMIUM": {
            "type": "accident_uniquement", 
            "taux_remboursement": 100,
            "plafond": 500,
            "description": "Fonds d'urgence accident – prise en charge des frais consécutifs à un accident, jusqu'à 500€ par an",
            "couverture": "Accidents uniquement",
            "color": "#FF6B35",
            "emoji": "🚨"
        },
        "INTEGRAL": {
            "type": "accident_et_maladie",
            "taux_remboursement": 50,
            "plafond": 1000,
            "description": "Assurance santé animale – prise en charge à 50% des frais vétérinaires (accident & maladie), plafond 1000€ par an",
            "couverture": "Accidents et maladies",
            "color": "#4ECDC4",
            "emoji": "💚"
        },
        "INTEGRAL_PLUS": {
            "type": "accident_et_maladie",
            "taux_remboursement": 100,
            "plafond": 1000,
            "description": "Assurance santé animale – prise en charge à 100% des frais vétérinaires (accident & maladie), plafond 1000€ par an",
            "couverture": "Accidents et maladies",
            "color": "#A29BFE",
            "emoji": "💜"
        }
    }
    return formules_pennypet.get(formule, formules_pennypet["START"])

def get_pennypet_formule_card(formule, info):
    """Crée une carte de formule PennyPet"""
    return f"""
    <div class="pennypet-card formule-{formule.lower().replace('_', '-')}">
        <h4>{info['emoji']} {formule}</h4>
        <p><strong>Couverture:</strong> {info['couverture']}</p>
        <p><strong>Remboursement:</strong> <span style="color: {info['color']}; font-weight: bold;">{info['taux_remboursement']}%</span></p>
        <p><strong>Plafond:</strong> <span style="color: {info['color']}; font-weight: bold;">{info['plafond']}€/an</span></p>
        <p style="font-style: italic; font-size: 0.9rem; margin-top: 1rem;">{info['description']}</p>
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

# Initialisation de la session
if 'processing_step' not in st.session_state:
    st.session_state.processing_step = 0
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'client_info' not in st.session_state:
    st.session_state.client_info = None

# En-tête PennyPet avec gradients rose-violet
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("""
<div class="pennypet-animated">
    <h1 class="pennypet-header">🐾 PennyPet Remboursements</h1>
    <p class="pennypet-subtitle">
        <strong>Ton assistant intelligent pour les remboursements vétérinaires</strong><br>
        💚 Ensemble, prenons soin de nos compagnons à quatre pattes
    </p>
</div>
""", unsafe_allow_html=True)

# Connexion Supabase
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

# Sidebar PennyPet avec gradients rose-violet
with st.sidebar:
    st.markdown("### 🛠️ Configuration PennyPet")
    
    # Mascotte PennyPet
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 12px; margin-bottom: 1rem; backdrop-filter: blur(10px);">
        <h3 style="color: white; margin: 0;">🐕 PennyPet</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">Ton compagnon remboursement !</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🤖 Assistant IA", expanded=True):
        provider = st.selectbox(
            "Modèle d'extraction",
            ["qwen", "mistral"],
            index=0,
            help="🧠 Choisis ton assistant IA PennyPet pour analyser tes factures"
        )
        
        formules_possibles = ["START", "PREMIUM", "INTEGRAL", "INTEGRAL_PLUS"]
        formule_simulation = st.selectbox(
            "Formule (simulation)",
            formules_possibles,
            index=0,
            help="📋 Formule utilisée en mode simulation"
        )
    
    with st.expander("📋 Formules PennyPet", expanded=True):
        for formule in formules_possibles:
            info = get_pennypet_formule_info(formule)
            st.markdown(get_pennypet_formule_card(formule, info), unsafe_allow_html=True)
    
    # Communauté PennyPet
    st.markdown("### 🌟 Communauté PennyPet")
    st.markdown("""
    <div class="pennypet-card">
        <p style="text-align: center; margin: 0;">
            🐾 <strong>Plus de 11 000 pet-parents</strong><br>
            font confiance à PennyPet !
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistiques si disponibles
    if st.session_state.extraction_result:
        st.markdown("### 📊 Statistiques")
        stats = st.session_state.extraction_result.get('stats', {})
        st.metric("🔍 Lignes traitées", stats.get('lignes_traitees', 0))
        st.metric("💊 Médicaments détectés", stats.get('medicaments_detectes', 0))
        st.metric("🏥 Actes détectés", stats.get('actes_detectes', 0))

# Interface principale avec tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload & Extraction", "🔍 Identification Client", "💰 Remboursement", "📈 Analyse"])

with tab1:
    st.markdown("### 📁 Upload de ta Facture Vétérinaire")
    
    display_pennypet_alert(PENNYPET_MESSAGES["welcome"], "info", "🐾")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="pennypet-upload">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "🐕 Glisse-dépose ta facture ou clique pour sélectionner",
            type=["pdf", "jpg", "png", "jpeg"],
            label_visibility="collapsed",
            help="📋 Formats supportés: PDF, JPG, PNG (max 10MB)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded:
            is_valid, message = validate_file(uploaded)
            if is_valid:
                display_pennypet_alert(
                    f"✅ {message} - {uploaded.name} ({uploaded.size:,} bytes)", 
                    "success", 
                    "🎉"
                )
            else:
                display_pennypet_alert(f"❌ {message}", "error", "😔")
                st.stop()
    
    with col2:
        if uploaded:
            if uploaded.type.startswith('image/'):
                try:
                    image = Image.open(uploaded)
                    st.image(image, caption="📸 Prévisualisation", use_column_width=True)
                except Exception as e:
                    display_pennypet_alert(f"Erreur prévisualisation: {e}", "warning", "⚠️")
            else:
                st.markdown("""
                <div class="pennypet-card">
                    <h4>📄 PDF détecté</h4>
                    <p>Le fichier sera automatiquement converti en image pour l'analyse par notre IA PennyPet.</p>
                </div>
                """, unsafe_allow_html=True)
    
    if uploaded and st.button("🚀 Analyser ma facture avec PennyPet", type="primary"):
        bytes_data = uploaded.read()
        if not bytes_data:
            display_pennypet_alert("⚠️ Le fichier est vide ou corrompu.", "error", "😔")
            st.stop()
        
        processor = PennyPetProcessor()
        progress_placeholder = st.empty()
        
        progress_placeholder.markdown(
            create_pennypet_progress(1, 3, "Extraction des données..."), 
            unsafe_allow_html=True
        )
        
        try:
            with st.spinner("🔍 L'IA PennyPet analyse ta facture..."):
                temp = processor.process_facture_pennypet(
                    file_bytes=bytes_data,
                    formule_client="INTEGRAL",
                    llm_provider=provider
                )
                
                if not temp.get('success', False):
                    display_pennypet_alert(
                        f"❌ {PENNYPET_MESSAGES['error']}: {temp.get('error', 'Erreur inconnue')}", 
                        "error", 
                        "😔"
                    )
                    st.stop()
                
                st.session_state.extraction_result = temp
                st.session_state.processing_step = 1
                
        except Exception as e:
            display_pennypet_alert(
                f"❌ {PENNYPET_MESSAGES['error']}: {str(e)}", 
                "error", 
                "😔"
            )
            st.stop()
        
        progress_placeholder.markdown(
            create_pennypet_progress(3, 3, "Extraction terminée !"), 
            unsafe_allow_html=True
        )
        display_pennypet_alert(PENNYPET_MESSAGES["upload_success"], "success", "🎉")

with tab2:
    st.markdown("### 🔍 Identification de ton Contrat PennyPet")
    
    if not st.session_state.extraction_result:
        display_pennypet_alert(
            "⚠️ Commence par extraire les données de ta facture dans l'onglet précédent.", 
            "warning", 
            "💡"
        )
    else:
        display_pennypet_alert(PENNYPET_MESSAGES["search_client"], "info", "🔍")
        
        # Récupération des informations client
        infos = st.session_state.extraction_result.get("informations_client", {})
        identification = infos.get("identification")
        nom_proprietaire = infos.get("nom_proprietaire")
        nom_animal = infos.get("nom_animal")
        
        st.markdown("#### 📋 Informations extraites de ta facture")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="pennypet-card pennypet-animated">
                <h4>👤 Propriétaire</h4>
                <p><strong>{nom_proprietaire or 'Non détecté'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="pennypet-card pennypet-animated">
                <h4>🐾 Animal</h4>
                <p><strong>{nom_animal or 'Non détecté'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="pennypet-card pennypet-animated">
                <h4>🆔 Identification</h4>
                <p><strong>{identification or 'Non détecté'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 🔗 Recherche de ton contrat PennyPet")
        
        search_performed = False
        res = []
        
        if st.button("🔍 Rechercher mon contrat PennyPet", type="primary"):
            search_performed = True
            
            with st.spinner("🔍 Recherche dans la base PennyPet..."):
                try:
                    # Recherche par identification
                    if identification:
                        with st.expander("🆔 Recherche par identification", expanded=True):
                            st.info(f"🔍 Recherche pour: {identification}")
                            res = supabase.table("contrats_animaux") \
                                .select("proprietaire,animal,type_animal,date_naissance,identification,formule") \
                                .eq("identification", identification) \
                                .limit(1).execute().data
                            
                            if res:
                                display_pennypet_alert("✅ Contrat trouvé par identification!", "success", "🎉")
                            else:
                                st.warning("⚠️ Aucun contrat trouvé par identification")
                    
                    # Fallback par nom de propriétaire
                    if not res and nom_proprietaire:
                        with st.expander("👤 Recherche par nom de propriétaire", expanded=True):
                            terme = f"%{nom_proprietaire.strip()}%"
                            st.info(f"🔍 Recherche pour: {terme}")
                            rpc_resp = supabase.rpc("search_contrat_by_name", {"term": terme}).execute()
                            res = rpc_resp.data
                            
                            if res:
                                display_pennypet_alert(f"✅ {len(res)} contrat(s) trouvé(s) par nom!", "success", "🎉")
                            else:
                                st.warning("⚠️ Aucun contrat trouvé par nom de propriétaire")
                    
                    # Fallback par nom d'animal
                    if not res and nom_animal:
                        with st.expander("🐾 Recherche par nom d'animal", expanded=True):
                            terme = f"%{nom_animal.strip()}%"
                            st.info(f"🔍 Recherche pour: {terme}")
                            rpc_resp = supabase.rpc("search_contrat_by_name", {"term": terme}).execute()
                            res = rpc_resp.data
                            
                            if res:
                                display_pennypet_alert(f"✅ {len(res)} contrat(s) trouvé(s) par nom d'animal!", "success", "🎉")
                            else:
                                st.warning("⚠️ Aucun contrat trouvé par nom d'animal")
                                
                except Exception as e:
                    display_pennypet_alert(f"❌ Erreur lors de la recherche: {str(e)}", "error", "😔")
        
        # Gestion des résultats de recherche
        if search_performed:
            if res and len(res) == 1:
                client = res[0]
                formule_client = client["formule"]
                st.session_state.client_info = client
                
                st.markdown("#### ✅ Ton contrat PennyPet identifié")
                
                info_formule = get_pennypet_formule_info(formule_client)
                st.markdown(f"""
                <div class="pennypet-card pennypet-animated">
                    <h4>📋 Informations de ton contrat</h4>
                    <p><strong>Propriétaire:</strong> {client['proprietaire']}</p>
                    <p><strong>Animal:</strong> {client['animal']} ({client.get('type_animal', 'Non spécifié')})</p>
                    <p><strong>Identification:</strong> {client['identification']}</p>
                    <div style="background: {info_formule['color']}; color: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <h5>{info_formule['emoji']} Formule {client['formule']}</h5>
                        <p><strong>Couverture:</strong> {info_formule['couverture']}</p>
                        <p><strong>Remboursement:</strong> {info_formule['taux_remboursement']}%</p>
                        <p><strong>Plafond:</strong> {info_formule['plafond']}€/an</p>
                        <p><em>{info_formule['description']}</em></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                display_pennypet_alert("🎉 Parfait ! Passe à l'onglet Remboursement pour voir tes gains.", "success", "🚀")
                
            elif res and len(res) > 1:
                st.markdown("#### 🔄 Plusieurs contrats PennyPet trouvés")
                
                options = [f"{r['proprietaire']} – {r['animal']} ({r['identification']})" for r in res]
                choix = st.selectbox("🎯 Sélectionne ton contrat approprié:", options)
                
                if choix:
                    idx = options.index(choix)
                    client = res[idx]
                    formule_client = client["formule"]
                    st.session_state.client_info = client
                    
                    info_formule = get_pennypet_formule_info(formule_client)
                    st.markdown(f"""
                    <div class="pennypet-card pennypet-animated">
                        <h4>📋 Ton contrat sélectionné</h4>
                        <p><strong>Propriétaire:</strong> {client['proprietaire']}</p>
                        <p><strong>Animal:</strong> {client['animal']} ({client.get('type_animal', 'Non spécifié')})</p>
                        <div style="background: {info_formule['color']}; color: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                            <h5>{info_formule['emoji']} Formule {client['formule']}</h5>
                            <p><strong>Couverture:</strong> {info_formule['couverture']}</p>
                            <p><strong>Remboursement:</strong> {info_formule['taux_remboursement']}%</p>
                            <p><strong>Plafond:</strong> {info_formule['plafond']}€/an</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    display_pennypet_alert("✅ Super ! Passe à l'onglet Remboursement.", "success", "🚀")
            else:
                st.markdown("#### 💡 Mode Simulation PennyPet")
                display_pennypet_alert(PENNYPET_MESSAGES["no_contract"], "warning", "💡")
                
                with st.form("simulation_form"):
                    st.markdown("##### 📝 Informations pour la simulation")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        sim_proprietaire = st.text_input("👤 Ton nom", value=nom_proprietaire or "")
                        sim_animal = st.text_input("🐾 Nom de ton animal", value=nom_animal or "")
                    
                    with col2:
                        sim_type = st.selectbox("🏠 Type d'animal", ["Chien", "Chat", "NAC", "Autre"])
                        sim_formule = st.selectbox("📋 Formule à simuler", formules_possibles)
                    
                    if st.form_submit_button("💻 Lancer la simulation PennyPet"):
                        client = {
                            "proprietaire": sim_proprietaire or "Simulation",
                            "animal": sim_animal or "Simulation",
                            "type_animal": sim_type,
                            "formule": sim_formule,
                            "identification": "SIMULATION"
                        }
                        st.session_state.client_info = client
                        
                        display_pennypet_alert("✅ Mode simulation activé ! Découvre tes remboursements PennyPet.", "success", "🎯")

with tab3:
    st.markdown("### 💰 Calcul de tes Remboursements PennyPet")
    
    if not st.session_state.extraction_result or not st.session_state.client_info:
        display_pennypet_alert("⚠️ Termine d'abord l'extraction et l'identification dans les onglets précédents.", "warning", "📋")
    else:
        client = st.session_state.client_info
        formule_client = client["formule"]
        
        st.markdown("#### 👤 Ton Profil PennyPet")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="pennypet-card">
                <h4>📋 Détails de ton contrat</h4>
                <p><strong>Propriétaire:</strong> {client['proprietaire']}</p>
                <p><strong>Animal:</strong> {client['animal']} ({client.get('type_animal', 'Non spécifié')})</p>
                <p><strong>Formule:</strong> <span style="color: var(--pennypet-primary); font-weight: bold;">{client['formule']}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            info_formule = get_pennypet_formule_info(formule_client)
            st.markdown(f"""
            <div class="pennypet-card">
                <div style="background: {info_formule['color']}; color: white; padding: 1rem; border-radius: 8px;">
                    <h4>{info_formule['emoji']} Couverture {formule_client}</h4>
                    <p><strong>Type:</strong> {info_formule['couverture']}</p>
                    <p><strong>Remboursement:</strong> {info_formule['taux_remboursement']}%</p>
                    <p><strong>Plafond annuel:</strong> {info_formule['plafond']}€</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("💳 Calculer mes remboursements PennyPet", type="primary"):
            processor = PennyPetProcessor()
            
            with st.spinner("💰 Calcul de tes remboursements PennyPet en cours..."):
                try:
                    bytes_data = uploaded.read() if uploaded else None
                    if not bytes_data:
                        display_pennypet_alert("❌ Données de facture non disponibles", "error", "😔")
                        st.stop()
                    
                    result = processor.process_facture_pennypet(
                        file_bytes=bytes_data,
                        formule_client=formule_client,
                        llm_provider=provider
                    )
                    
                    if not result.get('success', False):
                        display_pennypet_alert(f"❌ Calcul échoué: {result.get('error', 'Erreur inconnue')}", "error", "😔")
                        st.stop()
                    
                    st.session_state.remboursement_result = result
                    
                    display_pennypet_alert(PENNYPET_MESSAGES["calculation_done"], "success", "🎉")
                    
                    st.markdown("#### 📊 Tes Remboursements PennyPet")
                    
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
                                total_rembourse += 0
                            elif formule_client == "PREMIUM":
                                est_accident = item.get('est_accident', False)
                                if est_accident:
                                    remb = min(montant_ht, 500)
                                    total_rembourse += remb
                            elif formule_client == "INTEGRAL":
                                remb = min(montant_ht * 0.5, 1000)
                                total_rembourse += remb
                            elif formule_client == "INTEGRAL_PLUS":
                                remb = min(montant_ht, 1000)
                                total_rembourse += remb
                    
                    reste_a_charge = total_facture - total_rembourse
                    
                    # Affichage des métriques PennyPet
                    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
                    st.markdown(display_pennypet_metric(f"{total_facture:.2f}€", "Total Facture", "🧾"), unsafe_allow_html=True)
                    st.markdown(display_pennypet_metric(f"{total_rembourse:.2f}€", "Remboursé PennyPet", "💰"), unsafe_allow_html=True)
                    st.markdown(display_pennypet_metric(f"{reste_a_charge:.2f}€", "Reste à ta Charge", "💳"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tableau détaillé
                    st.markdown("#### 📋 Détail par ligne de ta facture")
                    
                    if result.get('lignes'):
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
                                'Remboursé PennyPet': f"{remb_individuel:.2f}€",
                                'Reste': f"{reste_individuel:.2f}€"
                            })
                        
                        df = pd.DataFrame(lignes_data)
                        st.dataframe(df, use_container_width=True)
                    
                    # Explication des règles PennyPet
                    st.markdown("#### 📖 Règles PennyPet appliquées")
                    info_formule = get_pennypet_formule_info(formule_client)
                    
                    if formule_client == "START":
                        display_pennypet_alert("📱 **Formule START**: Pas d'assurance incluse - Découvre nos autres formules !", "info", "💡")
                    elif formule_client == "PREMIUM":
                        display_pennypet_alert("🚨 **Formule PREMIUM**: Fonds d'urgence pour accidents uniquement - Remboursement à 100% jusqu'à 500€/an", "info", "🚨")
                    elif formule_client == "INTEGRAL":
                        display_pennypet_alert("💚 **Formule INTEGRAL**: Assurance santé animale - Remboursement à 50% pour accidents et maladies jusqu'à 1000€/an", "info", "💚")
                    elif formule_client == "INTEGRAL_PLUS":
                        display_pennypet_alert("💜 **Formule INTEGRAL_PLUS**: Assurance santé animale - Remboursement à 100% pour accidents et maladies jusqu'à 1000€/an", "info", "💜")
                    
                    # Bouton d'enregistrement
                    if client.get('identification') != 'SIMULATION':
                        if st.button("💾 Enregistrer mon remboursement PennyPet", type="secondary"):
                            try:
                                contrat_data = supabase.table("contrats_animaux") \
                                    .select("id") \
                                    .eq("identification", client["identification"]) \
                                    .limit(1).execute().data
                                
                                if contrat_data:
                                    contrat_id = contrat_data[0]["id"]
                                    
                                    supabase.table("remboursements").insert([{
                                        "id_contrat": contrat_id,
                                        "date_acte": datetime.now().isoformat(),
                                        "montant_facture": total_facture,
                                        "montant_rembourse": total_rembourse,
                                        "reste_a_charge": reste_a_charge
                                    }]).execute()
                                    
                                    display_pennypet_alert(PENNYPET_MESSAGES["save_success"], "success", "💾")
                                else:
                                    display_pennypet_alert("❌ Impossible de trouver le contrat", "error", "😔")
                                    
                            except Exception as e:
                                display_pennypet_alert(f"❌ Erreur lors de l'enregistrement: {str(e)}", "error", "😔")
                    
                except Exception as e:
                    display_pennypet_alert(f"❌ Erreur lors du calcul: {str(e)}", "error", "😔")

with tab4:
    st.markdown("### 📈 Analyse et Insights PennyPet")
    
    if not st.session_state.get('remboursement_result'):
        display_pennypet_alert("⚠️ Calcule d'abord tes remboursements pour voir l'analyse complète.", "warning", "📊")
    else:
        result = st.session_state.remboursement_result
        client = st.session_state.client_info
        
        st.markdown("#### 📊 Répartition de tes Remboursements PennyPet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calcul des totaux
            total_facture = sum(item.get('ligne', {}).get('montant_ht', 0) for item in result.get('lignes', []))
            total_rembourse = 0
            
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
                # Couleurs PennyPet pour les graphiques
                colors = ['#4ECDC4', '#FF6B35']
                
                fig_pie = px.pie(
                    values=[total_rembourse, reste_a_charge],
                    names=['Remboursé PennyPet', 'Reste à ta charge'],
                    title="💰 Répartition selon PennyPet",
                    color_discrete_sequence=colors
                )
                fig_pie.update_layout(height=400, font_family="Inter")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            if result.get('lignes'):
                medicaments_
