import plotly.express as px
import streamlit as st

from loaders.llama_index_setup import query_data
from utils.utils import download_excel, generate_page_prompt


def grant_amount_distribution(df, grouped_df, selected_chart, selected_role, ai_enabled):
    """
       Display and interact with the grant amount distribution visualization.

       Args:
           df (pd.DataFrame): The original dataset DataFrame.
           grouped_df (pd.DataFrame): The grouped dataset DataFrame.
           selected_chart (str): The selected chart type.
           selected_role (str): The selected user role.
           ai_enabled (bool): Flag indicating whether AI-powered features are enabled.

       Returns:
           None
       """

    # Display the header and description
    st.header("Grant Amount Distribution w AI Chat")
    st.write("""
        Dive into the dynamic landscape of grant funding with our interactive distribution chart. This tool lets you visualize how grants are dispersed across various USD clusters, offering a clear view of funding trends and concentrations. Select different clusters to tailor the data shown and discover patterns at a glance.
        """)

    # Display visualizations
    cluster_options = grouped_df['amount_usd_cluster'].unique().tolist()
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options)
    filtered_df = grouped_df[grouped_df['amount_usd_cluster'].isin(selected_clusters)]
    fig = px.bar(filtered_df, x='amount_usd_cluster', y='amount_usd', color='amount_usd_cluster',
                 title="Grant Amount Distribution by USD Cluster")
    st.plotly_chart(fig)

    if ai_enabled:

        # AI Query Interface
        st.subheader("Data Exploration with GPT-4 Assistant")
        st.write("""
                Unleash the power of AI to delve deeper into the data. Ask your questions using natural language and let GPT-4 AI assist you in uncovering nuanced insights and trends from the grant distribution chart.
                """)

        # Generate the custom prompt for the current page
        additional_context = "the distribution of grant amounts across different USD clusters"
        pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

        # Dropdown for predefined questions
        query_options = [
            "What is the average grant amount for each cluster?",
            "What are the key takeaways from the grant amount distribution chart?",
            "What are some other interesting insights we could make from this data?"
        ]
        selected_query = st.selectbox("Select a predefined question or choose 'Custom Question':",
                                      ["Custom Question"] + query_options)

        if selected_query == "Custom Question":
            # Allow users to enter their own question
            user_query = st.text_input("Enter your question here:")
            query_text = user_query
        else:
            query_text = selected_query

        # Button to submit query
        if st.button("Submit"):
            if query_text:
                response = query_data(filtered_df, query_text, pre_prompt)
                st.markdown(response)
            else:
                st.warning("Please enter a question or select a predefined question.")

    else:
        # Inform the user that AI features are disabled
        st.info("AI-assisted analysis is disabled. Please provide an API key to enable this feature.")

       # Button to download the data as an Excel file
    if st.button("Download Data for Chart"):
        download_excel(filtered_df, "grants_data_chart.xlsx")

    st.markdown(""" This app was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using the latest methods for enabling AI to Chat with Data. It also uses the Candid API, Streamlit, Plotly, and other open-source libraries. Generative AI solutions such as OpenAI GPT-4 and Claude Opus were used to generate portions of the source code.
                    """)
