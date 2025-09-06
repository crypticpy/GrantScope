import os, sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.app_state import init_session_state, sidebar_controls, get_data  # type: ignore
from plots.grant_amount_distribution import grant_amount_distribution  # type: ignore


st.set_page_config(page_title="GrantScope — Distribution", page_icon=":chart_with_upwards_trend:")

init_session_state()
uploaded_file, selected_role, ai_enabled = sidebar_controls()
df, grouped_df, err = get_data(uploaded_file)

st.title("GrantScope Dashboard")
if err:
    st.error(f"Data load error: {err}")
else:
    grant_amount_distribution(df, grouped_df, "Grant Amount Distribution", selected_role, ai_enabled)
