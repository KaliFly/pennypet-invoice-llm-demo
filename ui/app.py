import streamlit as st

st.title("🔍 Test de diagnostic PennyPet")

# Test 1 : Secrets
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    st.success("✅ Secrets Supabase OK")
except Exception as e:
    st.error(f"❌ Secrets Supabase : {e}")

# Test 2 : Secrets OpenRouter
try:
    api_key = st.secrets["openrouter"]["API_KEY_QWEN"]
    st.success("✅ Secrets OpenRouter OK")
except Exception as e:
    st.error(f"❌ Secrets OpenRouter : {e}")

# Test 3 : Import OpenRouter
try:
    from openrouter_client import OpenRouterClient
    st.success("✅ Import OpenRouterClient OK")
except Exception as e:
    st.error(f"❌ Import OpenRouterClient : {e}")

# Test 4 : Import Processor
try:
    from llm_parser.pennypet_processor import PennyPetProcessor
    st.success("✅ Import PennyPetProcessor OK")
except Exception as e:
    st.error(f"❌ Import PennyPetProcessor : {e}")
