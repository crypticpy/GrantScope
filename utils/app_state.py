import os
import streamlit as st
from loaders.data_loader import load_data, preprocess_data


def init_session_state():
    if 'cache_initialized' not in st.session_state:
        st.session_state.cache_initialized = False
    if not st.session_state.cache_initialized:
        st.cache_data.clear()
        st.session_state.cache_initialized = True


def sidebar_controls():
    st.sidebar.markdown("Click to contact me with any questions or feedback!")
    st.sidebar.markdown('<a href="mailto:dltzshz8@anonaddy.me">Contact the Developer!</a>', unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload Candid API JSON File 10MB or less", accept_multiple_files=False, type="json")

    api_key = os.getenv("OPENAI_API_KEY") or getattr(st, 'secrets', {}).get("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter your GPT-4 API Key:", type="password")
    ai_enabled = bool(api_key)
    if not ai_enabled:
        st.sidebar.warning("AI features are disabled. Provide an API key via environment, secrets, or input to enable AI-assisted analysis.")

    user_roles = ["Grant Analyst/Writer", "Normal Grant User"]
    selected_role = st.sidebar.selectbox("Select User Role", options=user_roles)

    return uploaded_file, selected_role, ai_enabled


@st.cache_data
def _load_and_preprocess(file_path: str | None, file_bytes):
    grants = load_data(file_path=file_path, uploaded_file=file_bytes)
    return preprocess_data(grants)


def get_data(uploaded_file):
    try:
        if uploaded_file is not None:
            df, grouped_df = _load_and_preprocess(None, uploaded_file)
        else:
            df, grouped_df = _load_and_preprocess('data/sample.json', None)
        return df, grouped_df, None
    except (OSError, ValueError, KeyError) as e:
        return None, None, str(e)
