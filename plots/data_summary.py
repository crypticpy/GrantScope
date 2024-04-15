import streamlit as st
import plotly.express as px
import pandas as pd

from utils.utils import generate_page_prompt
from loaders.llama_index_setup import query_data

def data_summary(df, grouped_df, selected_chart, selected_role, ai_enabled):
    if selected_chart == "Data Summary":
        st.header("Introduction")
        st.write("""
        Welcome to the GrantScope Tool! This powerful prototype application is designed to assist grant writers and analysts in navigating and extracting insights from a complex grant dataset. By leveraging the capabilities of this tool, you can identify potential funding opportunities, analyze trends, and gain valuable information.

        The application has select pages which have been enhanced with experimental features from Llama-Index's QueryPipeline for Pandas Dataframes, and OpenAI GPT-4 to provide you with additional insights and analysis. You can interact with the AI assistant to ask questions, generate summaries, and explore the data in a more intuitive manner.
        
        The preloaded dataset encompasses a small sample of grant data, including details about funders, recipients, grant amounts, subject areas, populations served, and more. With this tool, you can explore the data through interactive visualizations, filter and search for specific grants, and download relevant data for further analysis.

        """)

        if selected_role == "Grant Analyst/Writer" and selected_chart == "Data Summary":
            st.subheader("Dataset Overview")
            st.write(df.head())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Unique Grants", value=df['grant_key'].nunique())
        with col2:
            st.metric(label="Total Unique Funders", value=df['funder_name'].nunique())
        with col3:
            st.metric(label="Total Unique Recipients", value=df['recip_name'].nunique())

        st.subheader("Top Funders by Total Grant Amount")
        top_n = st.slider("Select the number of top funders to display", min_value=5, max_value=20, value=10, step=1)
        unique_df = df.drop_duplicates(subset='grant_key')
        top_funders = unique_df.groupby('funder_name')['amount_usd'].sum().nlargest(top_n).reset_index()

        fig = px.bar(top_funders, x='funder_name', y='amount_usd', title=f"Top {top_n} Funders by Total Grant Amount")
        fig.update_layout(xaxis_title='Funder Name', yaxis_title='Total Grant Amount (USD)')
        st.plotly_chart(fig)

        if ai_enabled:

            st.subheader("Top Funders Analysis with GPT-4 AI Assistant")
            st.write("Ask questions about the top funders and their grant amounts.")

            # Generate the custom prompt for the top funders chart
            additional_context = f"the top {top_n} funders by total grant amount"
            pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

            # Predefined questions for top funders
            query_options = [
                "Which funder has the highest total grant amount?",
                "What is the average grant amount for the top 5 funders?",
                "Are there any notable trends or patterns among the top funders?"
            ]

            selected_query = st.selectbox("Select a predefined question or choose 'Custom Question':",
                                          ["Custom Question"] + query_options, key="top_funders_query")

            if selected_query == "Custom Question":
                user_query = st.text_input("Enter your question here:", key="top_funders_user_query")
                query_text = user_query
            else:
                query_text = selected_query

            if st.button("Submit", key="top_funders_submit"):
                if query_text:
                    response = query_data(top_funders, query_text, pre_prompt)
                    st.markdown(response)
                else:
                    st.warning("Please enter a question or select a predefined question.")
        else:
            st.info("AI-assisted analysis is disabled. Please provide an API key to enable this feature.")

        if st.checkbox("Show Top Funders Data Table"):
            st.write(top_funders)

        st.subheader("Grant Distribution by Funder Type")
        funder_type_dist = unique_df.groupby('funder_type')['amount_usd'].sum().reset_index()
        # Check if there are more than 10 funder types
        if len(funder_type_dist) > 10:
            # Sort the dataframe by 'amount_usd' to ensure we're aggregating the smallest categories
            funder_type_dist_sorted = funder_type_dist.sort_values(by='amount_usd', ascending=False)
            # Create a new dataframe for top 11 categories
            top_categories = funder_type_dist_sorted.head(11).copy()
            # Calculate the sum of 'amount_usd' for the "Other" category
            other_sum = funder_type_dist_sorted.iloc[11:]['amount_usd'].sum()
            # Append the "Other" category to the top categories dataframe
            other_row = pd.DataFrame(data={'funder_type': ['Other'], 'amount_usd': [other_sum]})
            top_categories = pd.concat([top_categories, other_row], ignore_index=True)
        else:
            top_categories = funder_type_dist

        # Generate the pie chart with the possibly modified dataframe
        fig = px.pie(top_categories, values='amount_usd', names='funder_type',
                     title="Grant Distribution by Funder Type")
        st.plotly_chart(fig)

        # Display the table for the "Other" category if it exists
        if len(funder_type_dist) > 12:
            st.subheader("Details of 'Other' Funder Types")
            # Get the rows that were aggregated into "Other"
            other_details = funder_type_dist_sorted.iloc[11:].reset_index(drop=True)
            st.dataframe(other_details.style.format({'amount_usd': "{:,.2f}"}))

        if st.checkbox("Show Funder Type Data Table"):
            st.write(funder_type_dist)

        st.subheader("Grant Distribution by Subject Area")
        subject_dist = unique_df.groupby('grant_subject_tran')['amount_usd'].sum().nlargest(10).reset_index()

        fig = px.bar(subject_dist, x='grant_subject_tran', y='amount_usd',
                     title="Top 10 Grant Subject Areas by Total Amount")
        fig.update_layout(xaxis_title='Subject Area', yaxis_title='Total Grant Amount (USD)')
        st.plotly_chart(fig)

        st.subheader("Grant Distribution by Population Served")
        population_dist = unique_df.groupby('grant_population_tran')['amount_usd'].sum().nlargest(10).reset_index()

        fig = px.bar(population_dist, x='grant_population_tran', y='amount_usd',
                     title="Top 10 Populations Served by Total Grant Amount")
        fig.update_layout(xaxis_title='Population Served', yaxis_title='Total Grant Amount (USD)')
        st.plotly_chart(fig)

        if ai_enabled:

            st.subheader("General Analysis with GPT-4 Assistant")
            st.write("Ask general questions about the grant dataset.")

            # Generate the custom prompt for general data analysis
            additional_context = "the overall grant dataset, including funders, recipients, grant amounts, subject areas, and populations served"
            pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

            # Predefined questions for general data analysis
            query_options = [
                "What are the key insights from the grant dataset?",
                "How has the distribution of grant amounts changed over time?",
                "Are there any correlations between subject areas and populations served?"
            ]

            selected_query = st.selectbox("Select a predefined question or choose 'Custom Question':",
                                          ["Custom Question"] + query_options, key="general_query")

            if selected_query == "Custom Question":
                user_query = st.text_input("Enter your question here:", key="general_user_query")
                query_text = user_query
            else:
                query_text = selected_query

            if st.button("Submit", key="general_submit"):
                if query_text:
                    response = query_data(df, query_text, pre_prompt)
                    st.markdown(response)
                else:
                    st.warning("Please enter a question or select a predefined question.")
        else:
            st.info("AI-assisted analysis is disabled. Please provide an API key to enable this feature.")

        st.divider()

        st.write("""
        This page serves as a glimpse into insights you can uncover using GrantScope. Feel free to explore the other plots of the application by using the menu on the left. From there you can dive deeper into specific aspects of the grant data, such as analyzing trends over time or examining population distributions.

        Happy exploring and best of luck with your grant related endeavors!

        This app was produced with Candid API, Streamlit, Plotly, and other open-source libraries. Generative AI solution such as Open ai GPT-4 and Claude Opus were used to generate portions of the source code.
        """)