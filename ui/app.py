# ui/app.py

import sys, os
# 1) Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

st.title("🔍 PennyPet Startup Diagnostics")

# 2) Test secrets
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    st.success("✅ Supabase secrets present")
except Exception as e:
    st.error(f"❌ Supabase secrets error: {e}")
    st.stop()

try:
    or_qwen = st.secrets["openrouter"]["API_KEY_QWEN"]
    st.success("✅ OpenRouter QWEN secret present")
except Exception as e:
    st.error(f"❌ OpenRouter secret error: {e}")
    st.stop()

# 3) Test imports
try:
    import openrouter_client
    st.success("✅ openrouter_client import OK")
except Exception as e:
    st.error(f"❌ openrouter_client import failed: {e}")
    st.stop()

try:
    from llm_parser.pennypet_processor import PennyPetProcessor
    st.success("✅ PennyPetProcessor import OK")
except Exception as e:
    st.error(f"❌ PennyPetProcessor import failed: {e}")
    st.stop()

st.info("🎉 All startup checks passed—app code beyond this point can now run.")
