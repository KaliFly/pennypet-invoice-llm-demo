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
    .formule-start { border-left-color: #
