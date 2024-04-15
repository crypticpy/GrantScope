import streamlit as st
import os
import openai
from loaders.data_loader import load_data, preprocess_data
from plots.data_summary import data_summary
from plots.grant_amount_distribution import grant_amount_distribution
from plots.grant_amount_scatter_plot import grant_amount_scatter_plot
from plots.grant_amount_heatmap import grant_amount_heatmap
from plots.grant_description_word_clouds import grant_description_word_clouds
from plots.treemaps_extended_analysis import treemaps_extended_analysis
from plots.general_analysis_relationships import general_analysis_relationships
from plots.top_categories_unique_grants import top_categories_unique_grants

def clear_cache():
    st.cache_data.clear()

def init_session_state():
    if 'cache_initialized' not in st.session_state:
        st.session_state.cache_initialized = False

    if not st.session_state.cache_initialized:
        clear_cache()
        st.session_state.cache_initialized = True

uploaded_file = None

st.set_page_config(page_title="GrantScope", page_icon=":chart_with_upwards_trend:")


def main():
    init_session_state()

    file_path = 'old/fixed_ovp.json'
    uploaded_file = st.sidebar.file_uploader("Upload Candid API JSON File 10MB or less", accept_multiple_files=False, type="json")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        api_key = st.sidebar.text_input("Enter your GPT-4 API Key:", type="password")

    if not api_key:
        st.warning(
            "Please set the OPENAI_API_KEY environment variable or enter your GPT-4 API key to use the AI features.")
        return

    openai.api_key = api_key

    if uploaded_file is not None:
        grants = load_data(uploaded_file=uploaded_file)
    else:
        grants = load_data(file_path=file_path)

    df, grouped_df = preprocess_data(grants)

    chart_options = {
        "Grant Analyst/Writer": [
            "Data Summary",
            "Grant Amount Distribution",
            "Grant Amount Scatter Plot",
            "Grant Amount Heatmap",
            "Grant Description Word Clouds",
            "Treemaps with Extended Analysis",
            "General Analysis of Relationships",
            "Top Categories by Unique Grant Count"
        ],
        "Normal Grant User": [
            "Data Summary",
            "Grant Amount Distribution",
            "Grant Amount Scatter Plot",
            "Grant Amount Heatmap",
            "Grant Description Word Clouds",
            "Treemaps with Extended Analysis"
        ]
    }

    user_roles = ["Grant Analyst/Writer", "Normal Grant User"]
    selected_role = st.sidebar.selectbox("Select User Role", options=user_roles)
    selected_chart = st.sidebar.selectbox("Select Chart", options=chart_options[selected_role])

    st.title("GrantScope Dashboard")

    if selected_chart == "Data Summary":
        data_summary(df, grouped_df, selected_chart, selected_role)
    elif selected_chart == "Grant Amount Distribution":
        grant_amount_distribution(df, grouped_df, selected_chart, selected_role)
    elif selected_chart == "Grant Amount Scatter Plot":
        grant_amount_scatter_plot(df, grouped_df, selected_chart, selected_role)
    elif selected_chart == "Grant Amount Heatmap":
        grant_amount_heatmap(df, grouped_df, selected_chart, selected_role)
    elif selected_chart == "Grant Description Word Clouds":
        grant_description_word_clouds(df, grouped_df, selected_chart, selected_role)
    elif selected_chart == "Treemaps with Extended Analysis":
        treemaps_extended_analysis(df, grouped_df, selected_chart, selected_role)
    elif selected_chart == "General Analysis of Relationships":
        general_analysis_relationships(df, grouped_df, selected_chart, selected_role)
    elif selected_chart == "Top Categories by Unique Grant Count":
        top_categories_unique_grants(df, grouped_df, selected_chart, selected_role)

if __name__ == '__main__':
    main()