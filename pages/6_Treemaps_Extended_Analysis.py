import os, sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.app_state import init_session_state, sidebar_controls, get_data  # type: ignore
from plots.treemaps_extended_analysis import treemaps_extended_analysis  # type: ignore


st.set_page_config(page_title="GrantScope — Treemaps", page_icon=":evergreen_tree:")

init_session_state()
uploaded_file, selected_role, ai_enabled = sidebar_controls()
df, grouped_df, err = get_data(uploaded_file)

st.title("GrantScope Dashboard")
if err:
    st.error(f"Data load error: {err}")
else:
    treemaps_extended_analysis(df, grouped_df, "Treemaps with Extended Analysis", selected_role, ai_enabled)
