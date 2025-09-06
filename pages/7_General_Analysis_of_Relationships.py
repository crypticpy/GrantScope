import os, sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.app_state import init_session_state, sidebar_controls, get_data  # type: ignore
from plots.general_analysis_relationships import general_analysis_relationships  # type: ignore


st.set_page_config(page_title="GrantScope — Relationships", page_icon=":handshake:")

init_session_state()
uploaded_file, selected_role, ai_enabled = sidebar_controls()
df, grouped_df, err = get_data(uploaded_file)

st.title("GrantScope Dashboard")
if err:
    st.error(f"Data load error: {err}")
else:
    general_analysis_relationships(df, grouped_df, "General Analysis of Relationships", selected_role, ai_enabled)
