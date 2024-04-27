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
    st.sidebar.markdown("Click to contact me with any questions or feedback!")
    st.sidebar.markdown('<a href="mailto:dltzshz8@anonaddy.me">Contact the Developer!</a>', unsafe_allow_html=True)
    file_path = 'data/sample.json'
    uploaded_file = st.sidebar.file_uploader("Upload Candid API JSON File 10MB or less", accept_multiple_files=False, type="json")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        api_key = st.sidebar.text_input("Enter your GPT-4 API Key:", type="password")

    if api_key:
        openai.api_key = api_key
        ai_enabled = True
    else:
        ai_enabled = False
        st.sidebar.warning(
            "AI features are disabled. Please set the OPENAI_API_KEY environment variable or enter your GPT-4 API key to enable AI-assisted analysis.")

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
        data_summary(df, grouped_df, selected_chart, selected_role, ai_enabled)
    elif selected_chart == "Grant Amount Distribution":
        grant_amount_distribution(df, grouped_df, selected_chart, selected_role, ai_enabled)
    elif selected_chart == "Grant Amount Scatter Plot":
        grant_amount_scatter_plot(df, grouped_df, selected_chart, selected_role, ai_enabled)
    elif selected_chart == "Grant Amount Heatmap":
        grant_amount_heatmap(df, grouped_df, selected_chart, selected_role, ai_enabled)
    elif selected_chart == "Grant Description Word Clouds":
        grant_description_word_clouds(df, grouped_df, selected_chart, selected_role, ai_enabled)
    elif selected_chart == "Treemaps with Extended Analysis":
        treemaps_extended_analysis(df, grouped_df, selected_chart, selected_role, ai_enabled)
    elif selected_chart == "General Analysis of Relationships":
        general_analysis_relationships(df, grouped_df, selected_chart, selected_role, ai_enabled)
    elif selected_chart == "Top Categories by Unique Grant Count":
        top_categories_unique_grants(df, grouped_df, selected_chart, selected_role, ai_enabled)

if __name__ == '__main__':
    main()