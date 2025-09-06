import os, sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.app_state import init_session_state, sidebar_controls, get_data  # type: ignore
from plots.grant_amount_scatter_plot import grant_amount_scatter_plot  # type: ignore


st.set_page_config(page_title="GrantScope — Scatter Plot", page_icon=":small_blue_diamond:")

init_session_state()
uploaded_file, selected_role, ai_enabled = sidebar_controls()
df, grouped_df, err = get_data(uploaded_file)

st.title("GrantScope Dashboard")
if err:
    st.error(f"Data load error: {err}")
else:
    grant_amount_scatter_plot(df, grouped_df, "Grant Amount Scatter Plot", selected_role, ai_enabled)
