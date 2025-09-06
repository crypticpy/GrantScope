import os, sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.app_state import init_session_state, sidebar_controls, get_data  # type: ignore
from plots.top_categories_unique_grants import top_categories_unique_grants  # type: ignore


st.set_page_config(page_title="GrantScope — Top Categories", page_icon=":top:")

init_session_state()
uploaded_file, selected_role, ai_enabled = sidebar_controls()
df, grouped_df, err = get_data(uploaded_file)

st.title("GrantScope Dashboard")
if err:
    st.error(f"Data load error: {err}")
else:
    top_categories_unique_grants(df, grouped_df, "Top Categories by Unique Grant Count", selected_role, ai_enabled)
