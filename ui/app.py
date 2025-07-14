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
from llm_parser.pennypet_processor import PennyPetProcessor, parse_llm_json

# Configuration de la page
st.set_page_config(
    page_title="PennyPet – Remboursements Vétérinaires",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS avec charte PennyPet (fond sobre + gradients rose-violet)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --pennypet-primary: #FF6B35;
        --pennypet-secondary: #4ECDC4;
        --pennypet-accent: #45B7D1;
        --pennypet-success: #96CEB4;
        --pennypet-warning: #FFEAA7;
        --pennypet-purple: #A29BFE;
        --pennypet-pink: #fd79a8;
        --pennypet-dark: #2D3436;
        --pennypet-light: #F8F9FA;
        --pennypet-gradient: linear-gradient(135deg, #FF6B35 0%, #4ECDC4 100%);
        --pennypet-gradient-title: linear-gradient(135deg, #fd79a8 0%, #A29BFE 50%, #6c5ce7 100%);
        --pennypet-shadow: 0 8px 32px rgba(255, 107, 53, 0.1);
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* FOND SOBRE */
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%);
        min-height: 100vh;
    }

    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(10px);
    }

    /* TITRE avec gradients rose-violet */
    .pennypet-header {
        text-align: center;
        background: var(--pennypet-gradient-title);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .pennypet-subtitle {
        text-align: center;
        color: var(--pennypet-dark);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* Cartes */
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
    }

    /* Métriques */
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

    .pennypet-metric:hover { transform: scale(1.05); }

    .pennypet-metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    /* Boutons */
    .stButton > button {
        background: var(--pennypet-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4) !important;
    }

    /* Alertes */
    .pennypet-alert-success {
        background: linear-gradient(135deg, rgba(150, 206, 180, 0.1) 0%, rgba(255, 234, 167, 0.1) 100%);
        border: 1px solid rgba(150, 206, 180, 0.3);
        color: var(--pennypet-dark);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--pennypet-success);
    }

    .pennypet-alert-error {
        background: linear-gradient(135deg, rgba(253, 121, 168, 0.1) 0%, rgba(232, 67, 147, 0.1) 100%);
        border: 1px solid rgba(253, 121, 168, 0.3);
        color: var(--pennypet-dark);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--pennypet-pink);
    }

    .pennypet-alert-warning {
        background: linear-gradient(135deg, rgba(255, 234, 167, 0.1) 0%, rgba(255, 107, 53, 0.1) 100%);
        border: 1px solid rgba(255, 107, 53, 0.3);
        color: var(--pennypet-dark);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--pennypet-warning);
    }

    .pennypet-alert-info {
        background: linear-gradient(135deg, rgba(69, 183, 209, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
        border: 1px solid rgba(69, 183, 209, 0.3);
        color: var(--pennypet-dark);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--pennypet-accent);
    }

    /* Sidebar avec gradients */
    .css-1d391kg {
        background: linear-gradient(180deg, #fd79a8 0%, #A29BFE 50%, #6c5ce7 100%);
    }

    .sidebar .stSelectbox > label {
        color: white !important;
        font-weight: 600 !important;
    }

    /* Upload zone */
    .pennypet-upload {
        border: 2px dashed var(--pennypet-primary);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(255, 245, 243, 0.5) 0%, rgba(240, 253, 252, 0.5) 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(248, 249, 250, 0.8);
        padding: 0.5rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: var(--pennypet-dark);
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: var(--pennypet-gradient-title);
        color: white;
        box-shadow: 0 4px 15px rgba(253, 121, 168, 0.3);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in { animation: fadeIn 0.6s ease-out; }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Messages PennyPet
PENNYPET_MESSAGES = {
    "welcome": "🐾 Bienvenue sur PennyPet ! Ensemble, simplifions la gestion des soins de ton animal",
    "upload_success": "✅ Super ! Ta facture a été analysée avec succès. Découvrons ensemble ce que PennyPet peut rembourser",
    "calculation_done": "🎉 Parfait ! Voici le détail de tes remboursements PennyPet",
    "save_success": "💾 Excellent ! Ton remboursement a été enregistré. L'équipe PennyPet s'occupe du reste !",
    "error": "😔 Oups ! Une petite erreur s'est glissée. L'équipe PennyPet va régler ça rapidement"
}

# Fonctions utilitaires
def display_pennypet_alert(message, alert_type="info", emoji="✨"):
    st.markdown(f'<div class="pennypet-alert-{alert_type} fade-in">{emoji} {message}</div>', unsafe_allow_html=True)

def display_pennypet_metric(value, label, emoji="📊"):
    return f"""
    <div class="pennypet-metric fade-in">
        <div class="pennypet-metric-value">{emoji} {value}</div>
        <div class="pennypet-metric-label" style="font-size: 0.9rem; opacity: 0.9;">{label}</div>
    </div>
    """

def get_pennypet_formule_info(formule):
    formules_pennypet = {
        "START": {"taux_remboursement": 0, "plafond": 0, "description": "Pas d'assurance incluse", "couverture": "Aucune couverture", "color": "#6c757d", "emoji": "📱"},
        "PREMIUM": {"taux_remboursement": 100, "plafond": 500, "description": "Fonds d'urgence accident – prise en charge des frais consécutifs à un accident, jusqu'à 500€ par an", "couverture": "Accidents uniquement", "color": "#FF6B35", "emoji": "🚨"},
        "INTEGRAL": {"taux_remboursement": 50, "plafond": 1000, "description": "Assurance santé animale – prise en charge à 50% des frais vétérinaires (accident & maladie), plafond 1000€ par an", "couverture": "Accidents et maladies", "color": "#4ECDC4", "emoji": "💚"},
        "INTEGRAL_PLUS": {"taux_remboursement": 100, "plafond": 1000, "description": "Assurance santé animale – prise en charge à 100% des frais vétérinaires (accident & maladie), plafond 1000€ par an", "couverture": "Accidents et maladies", "color": "#A29BFE", "emoji": "💜"}
    }
    return formules_pennypet.get(formule, formules_pennypet["START"])

def validate_file(uploaded_file):
    if not uploaded_file:
        return False, "Aucun fichier sélectionné"
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "Fichier trop volumineux (max 10MB)"
    if uploaded_file.type not in ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']:
        return False, "Type de fichier non supporté"
    return True, "Fichier valide"

# Initialisation session
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'client_info' not in st.session_state:
    st.session_state.client_info = None
if 'remboursement_result' not in st.session_state:
    st.session_state.remboursement_result = None

# En-tête
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("""
<div class="fade-in">
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
    if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
        st.error("❗ Veuillez définir SUPABASE_URL et SUPABASE_KEY dans vos secrets.")
        st.stop()
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# Sidebar
with st.sidebar:
    st.markdown("### 🛠️ Configuration PennyPet")
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 12px; margin-bottom: 1rem;">
        <h3 style="color: black; margin: 0;">🐕 PennyPet</h3>
        <p style="color: black; margin: 0; font-size: 0.9rem;">Ton compagnon remboursement !</p>
    </div>
    """, unsafe_allow_html=True)
    
    # DEBUG ACTIVÉ DÈS LE DÉBUT
    debug_enabled = st.checkbox("🔧 Activer mode debug", value=True)
    
    with st.expander("🤖 Assistant IA", expanded=True):
        provider = st.selectbox("Modèle d'extraction", ["qwen", "mistral"], index=0)
        formules_possibles = ["START", "PREMIUM", "INTEGRAL", "INTEGRAL_PLUS"]
        formule_simulation = st.selectbox("Formule (simulation)", formules_possibles, index=0)
    
    # Stats en temps réel
    if st.session_state.extraction_result:
        st.markdown("### 📊 Statistiques")
        stats = st.session_state.extraction_result.get('statistiques', {})
        st.metric("🔍 Lignes traitées", stats.get('lignes_traitees', 0))
        st.metric("💊 Médicaments détectés", stats.get('medicaments_detectes', 0))
        st.metric("🏥 Actes détectés", stats.get('actes_detectes', 0))

# TABS avec debug conditionnel
if debug_enabled:
    tab1, tab2, tab3, tab4, tab_debug = st.tabs([
        "📄 Upload & Extraction", 
        "🔍 Identification Client", 
        "💰 Remboursement", 
        "📈 Analyse",
        "🔧 Debug JSON"
    ])
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Upload & Extraction", 
        "🔍 Identification Client", 
        "💰 Remboursement", 
        "📈 Analyse"
    ])

with tab1:
    st.markdown("### 📁 Upload de ta Facture Vétérinaire")
    display_pennypet_alert(PENNYPET_MESSAGES["welcome"], "info", "🐾")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="pennypet-upload">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "🐕 Glisse-dépose ta facture ou clique pour sélectionner",
            type=["pdf", "jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded:
            is_valid, message = validate_file(uploaded)
            if is_valid:
                display_pennypet_alert(f"✅ {message} - {uploaded.name}", "success", "🎉")
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
                    <p>Le fichier sera automatiquement converti en image pour l'analyse.</p>
                </div>
                """, unsafe_allow_html=True)
    
    if uploaded and st.button("🚀 Analyser ma facture avec PennyPet", type="primary"):
        bytes_data = uploaded.read()
        if not bytes_data:
            display_pennypet_alert("⚠️ Le fichier est vide ou corrompu.", "error", "😔")
            st.stop()
        
        processor = PennyPetProcessor()
        
        try:
            with st.spinner("🔍 L'IA PennyPet analyse ta facture..."):
                result = processor.process_facture_pennypet(
                    file_bytes=bytes_data,
                    formule_client="INTEGRAL",
                    llm_provider=provider
                )
                
                if not result.get('success', False):
                    display_pennypet_alert(
                        f"❌ {PENNYPET_MESSAGES['error']}: {result.get('error', 'Erreur inconnue')}", 
                        "error", "😔"
                    )
                    # Afficher l'erreur détaillée en debug
                    if debug_enabled:
                        st.error(f"Détail erreur: {result.get('error', 'N/A')}")
                    st.stop()
                
                st.session_state.extraction_result = result
                
        except Exception as e:
            display_pennypet_alert(f"❌ {PENNYPET_MESSAGES['error']}: {str(e)}", "error", "😔")
            if debug_enabled:
                st.exception(e)
            st.stop()
        
        display_pennypet_alert(PENNYPET_MESSAGES["upload_success"], "success", "🎉")

with tab2:
    st.markdown("### 🔍 Identification de ton Contrat PennyPet")
    
    if not st.session_state.extraction_result:
        display_pennypet_alert("⚠️ Commence par extraire les données de ta facture.", "warning", "💡")
    else:
        # Informations extraites
        infos = st.session_state.extraction_result.get("informations_client", {})
        identification = infos.get("identification")
        nom_proprietaire = infos.get("nom_proprietaire")
        nom_animal = infos.get("nom_animal")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="pennypet-card fade-in">
                <h4>👤 Propriétaire</h4>
                <p><strong>{nom_proprietaire or 'Non détecté'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="pennypet-card fade-in">
                <h4>🐾 Animal</h4>
                <p><strong>{nom_animal or 'Non détecté'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="pennypet-card fade-in">
                <h4>🆔 Identification</h4>
                <p><strong>{identification or 'Non détecté'}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recherche contrat (simulation pour démo)
        if st.button("🔍 Rechercher mon contrat PennyPet", type="primary"):
            # Simulation d'un contrat trouvé
            client = {
                "proprietaire": nom_proprietaire or "Demo Propriétaire",
                "animal": nom_animal or "Demo Animal",
                "type_animal": "Chien",
                "formule": "INTEGRAL",
                "identification": identification or "DEMO123"
            }
            st.session_state.client_info = client
            
            info_formule = get_pennypet_formule_info("INTEGRAL")
            st.markdown(f"""
            <div class="pennypet-card fade-in">
                <h4>✅ Contrat PennyPet trouvé !</h4>
                <p><strong>Propriétaire:</strong> {client['proprietaire']}</p>
                <p><strong>Animal:</strong> {client['animal']} ({client['type_animal']})</p>
                <div style="background: {info_formule['color']}; color: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <h5>{info_formule['emoji']} Formule {client['formule']}</h5>
                    <p><strong>Couverture:</strong> {info_formule['couverture']}</p>
                    <p><strong>Remboursement:</strong> {info_formule['taux_remboursement']}%</p>
                    <p><strong>Plafond:</strong> {info_formule['plafond']}€/an</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 💰 Calcul de tes Remboursements PennyPet")
    
    if not st.session_state.extraction_result or not st.session_state.client_info:
        display_pennypet_alert("⚠️ Termine d'abord l'extraction et l'identification.", "warning", "📋")
    else:
        client = st.session_state.client_info
        formule_client = client["formule"]
        
        if st.button("💳 Calculer mes remboursements PennyPet", type="primary"):
            result = st.session_state.extraction_result
            
            if result.get('success'):
                st.session_state.remboursement_result = result
                display_pennypet_alert(PENNYPET_MESSAGES["calculation_done"], "success", "🎉")
                
                # Totaux
                resume = result.get('resume', {})
                total_facture = resume.get('total_facture', 0)
                total_rembourse = resume.get('total_rembourse', 0)
                reste_a_charge = resume.get('reste_a_charge', 0)
                
                # Métriques
                st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
                st.markdown(display_pennypet_metric(f"{total_facture:.2f}€", "Total Facture", "🧾"), unsafe_allow_html=True)
                st.markdown(display_pennypet_metric(f"{total_rembourse:.2f}€", "Remboursé PennyPet", "💰"), unsafe_allow_html=True)
                st.markdown(display_pennypet_metric(f"{reste_a_charge:.2f}€", "Reste à ta Charge", "💳"), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Tableau détaillé
                if result.get('lignes'):
                    lignes_data = []
                    for item in result['lignes']:
                        ligne = item.get('ligne', {})
                        lignes_data.append({
                            'Description': ligne.get('description', 'Non spécifié'),
                            'Type': '💊 Médicament' if ligne.get('est_medicament') else '🏥 Acte',
                            'Montant HT': f"{ligne.get('montant_ht', 0):.2f}€",
                            'Remboursé': f"{item.get('montant_rembourse', 0):.2f}€",
                            'Reste': f"{item.get('montant_reste_charge', 0):.2f}€"
                        })
                    
                    df = pd.DataFrame(lignes_data)
                    st.dataframe(df, use_container_width=True)

with tab4:
    st.markdown("### 📈 Analyse et Insights PennyPet")
    
    if not st.session_state.get('remboursement_result'):
        display_pennypet_alert("⚠️ Calcule d'abord tes remboursements.", "warning", "📊")
    else:
        result = st.session_state.remboursement_result
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            resume = result.get('resume', {})
            total_rembourse = resume.get('total_rembourse', 0)
            reste_a_charge = resume.get('reste_a_charge', 0)
            
            if total_rembourse > 0 or reste_a_charge > 0:
                fig_pie = px.pie(
                    values=[total_rembourse, reste_a_charge],
                    names=['Remboursé PennyPet', 'Reste à charge'],
                    title="💰 Répartition des coûts",
                    color_discrete_sequence=['#4ECDC4', '#FF6B35']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            stats = result.get('statistiques', {})
            st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
            st.markdown(display_pennypet_metric(stats.get('medicaments_detectes', 0), "Médicaments", "💊"), unsafe_allow_html=True)
            st.markdown(display_pennypet_metric(stats.get('actes_detectes', 0), "Actes", "🏥"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# TAB DEBUG (conditionnel)
if debug_enabled:
    with tab_debug:
        st.markdown("### 🐞 Debug JSON LLM")
        
        if st.session_state.extraction_result:
            raw_response = st.session_state.extraction_result.get("raw_llm_response", "")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📝 Réponse LLM brute")
                st.text_area("Contenu complet", raw_response, height=300)
                
                # Analyse des accolades
                open_braces = raw_response.count('{')
                close_braces = raw_response.count('}')
                st.markdown(f"**Accolades:** `{{` = {open_braces}, `}}` = {close_braces}")
                
                if open_braces != close_braces:
                    st.error(f"⚠️ Déséquilibre des accolades ! Différence: {abs(open_braces - close_braces)}")
                else:
                    st.success("✅ Accolades équilibrées")
            
            with col2:
                st.markdown("#### 🧪 Test Parser JSON")
                
                if st.button("🔍 Tester le parsing JSON"):
                    try:
                        parsed_data = parse_llm_json(raw_response)
                        st.success("✅ JSON parsé avec succès !")
                        st.json(parsed_data)
                        
                        # Statistiques du parsing
                        nb_lignes = len(parsed_data.get('lignes', []))
                        st.info(f"📊 {nb_lignes} ligne(s) extraite(s)")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur parsing JSON:")
                        st.exception(e)
                        
                        # Aide au debug
                        error_str = str(e)
                        if "Expecting ',' delimiter" in error_str:
                            st.warning("💡 **Suggestion:** Problème de virgule manquante dans le JSON")
                        elif "Unterminated string" in error_str:
                            st.warning("💡 **Suggestion:** Chaîne de caractères non fermée")
                        elif "Expecting" in error_str:
                            st.warning("💡 **Suggestion:** Structure JSON malformée")
                
                # Outils de debug supplémentaires
                st.markdown("#### 🛠️ Outils de debug")
                
                if st.button("🔍 Analyser la structure JSON"):
                    # Recherche de patterns problématiques
                    problems = []
                    
                    # Vérifier les virgules manquantes communes
                    if '"}{"' in raw_response:
                        problems.append("Objects JSON collés sans virgule")
                    if '}[' in raw_response or ']{' in raw_response:
                        problems.append("Structures imbriquées mal délimitées")
                    
                    # Vérifier les guillemets
                    single_quotes = raw_response.count("'")
                    if single_quotes > 0:
                        problems.append(f"{single_quotes} guillemets simples détectés")
                    
                    if problems:
                        st.warning("⚠️ **Problèmes détectés:**")
                        for problem in problems:
                            st.write(f"• {problem}")
                    else:
                        st.success("✅ Aucun problème évident détecté")
        else:
            st.info("🔍 Aucune extraction disponible pour le debug. Uploadez et analysez d'abord une facture.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
---
<div style="text-align: center; padding: 2rem;">
    <div style="background: linear-gradient(135deg, #fd79a8 0%, #A29BFE 50%, #6c5ce7 100%); padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1rem;">
        <h3 style="margin: 0;">🐾 PennyPet</h3>
        <p style="margin: 0.5rem 0 0 0;">Ton compagnon remboursement vétérinaire</p>
    </div>
    <p style="margin: 0; font-size: 0.9rem; color: #6c757d;">
        💚 Développé avec amour par l'équipe PennyPet - Ensemble, prenons soin de nos animaux
    </p>
</div>
""", unsafe_allow_html=True)
