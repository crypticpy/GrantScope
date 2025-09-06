import os, sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.app_state import init_session_state, sidebar_controls, get_data  # type: ignore
from plots.grant_description_word_clouds import grant_description_word_clouds  # type: ignore


st.set_page_config(page_title="GrantScope — Word Clouds", page_icon=":cloud:")

init_session_state()
uploaded_file, selected_role, ai_enabled = sidebar_controls()
df, grouped_df, err = get_data(uploaded_file)

st.title("GrantScope Dashboard")
if err:
    st.error(f"Data load error: {err}")
else:
    grant_description_word_clouds(df, grouped_df, "Grant Description Word Clouds", selected_role, ai_enabled)
