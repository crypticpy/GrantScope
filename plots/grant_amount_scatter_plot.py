import pandas as pd
import plotly.express as px
import streamlit as st

from loaders.llama_index_setup import query_data
from utils.utils import download_csv, generate_page_prompt


def grant_amount_scatter_plot(df, grouped_df, selected_chart, selected_role, ai_enabled):
    if selected_chart == "Grant Amount Scatter Plot":
        st.header("Grant Amount Scatter Plot")
        st.write("""
        Welcome to the Grant Amount Scatter Plot page! This AI interactive visualization allows you to explore the distribution of grant amounts over time with support from GPT-4.
        
        This visualization makes it easy to spot trends and patterns in grant amounts across different USD clusters.

        The scatter plot dynamically updates based on the selected USD clusters and year range available in the data. Use the filters on the left to customize your view:

        1. Select the desired start and end years using the number input fields.
        2. Choose the USD clusters you want to include in the plot using the multiselect dropdown.
        3. Adjust the marker size and opacity using the sliders to enhance the visual representation.

        Hover over the data points to view details such as the grant key, description, and amount. You can also click on the legend items to toggle the visibility of specific USD clusters or double-click to isolate a single cluster.

        Feel free to download the underlying data as a CSV file for further analysis.
        """)

        unique_years = sorted(df['year_issued'].unique())
        if len(unique_years) == 1:
            unique_year = int(unique_years[0])
            st.write(f"Data available for year: {unique_year}")
            start_year, end_year = unique_year, unique_year
        else:
            start_year = st.number_input("Start Year", min_value=int(min(unique_years)),
                                         max_value=int(max(unique_years)), value=int(min(unique_years)))
            end_year = st.number_input("End Year", min_value=int(min(unique_years)), max_value=int(max(unique_years)),
                                       value=int(max(unique_years)))

        filtered_df = grouped_df[
            (grouped_df['year_issued'].astype(int) >= start_year) &
            (grouped_df['year_issued'].astype(int) <= end_year)
            ]
        filtered_df = filtered_df[filtered_df['amount_usd'].notna() & (filtered_df['amount_usd'] > 0)]

        cluster_options = filtered_df['amount_usd_cluster'].unique().tolist()
        selected_clusters = st.multiselect(
            "Select USD Clusters",
            options=cluster_options,
            default=cluster_options,
            key='scatter_clusters'
        )
        filtered_df = filtered_df[filtered_df['amount_usd_cluster'].isin(selected_clusters)]

        marker_size = st.slider("Marker Size", min_value=1, max_value=20, value=5)
        opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

        filtered_df['year_issued'] = pd.to_datetime(filtered_df['year_issued'], format='%Y')
        unique_years = filtered_df['year_issued'].dt.year.unique()
        unique_years.sort()

        fig = px.scatter(
            filtered_df,
            x='year_issued',
            y='amount_usd',
            color='amount_usd_cluster',
            hover_data=['grant_key', 'grant_description', 'amount_usd'],
            opacity=opacity
        )
        fig.update_traces(marker=dict(size=marker_size))
        fig.update_layout(
            title='Grant Amount by Year',
            xaxis_title='Year Issued',
            yaxis_title='Amount (USD)',
            legend_title_text='USD Cluster',
            legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"),
            clickmode='event+select'
        )
        fig.update_xaxes(
            tickvals=pd.to_datetime(unique_years, format='%Y'),
            ticktext=unique_years
        )

        st.plotly_chart(fig)

        if ai_enabled:

            # AI-Assisted Chat
            st.subheader("Scatter Plot Exploration with GPT-4 Assistant")
            st.write("Ask questions about the Grant Amount Scatter Plot to gain insights and explore the data further.")

            # Generate the custom prompt for the current page
            additional_context = f"the distribution of grant amounts over time, with data filtered by USD clusters ({', '.join(selected_clusters)}) and year range ({start_year} to {end_year})"
            pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

            # Predefined questions
            query_options = [
                "What are the overall trends in grant amounts over the selected time period?",
                "Which USD cluster has the highest average grant amount?",
                "Are there any outliers or unusual patterns in the scatter plot?",
                "How does the distribution of grant amounts vary across different years?",
                "What insights can we gain from the grant descriptions of the larger grant amounts?"
            ]

            selected_query = st.selectbox("Select a predefined question or choose 'Custom Question':",
                                          ["Custom Question"] + query_options)

            if selected_query == "Custom Question":
                # Allow users to enter their own question
                user_query = st.text_input("Enter your question here:")
                query_text = user_query
            else:
                query_text = selected_query

            # Button to submit the query
            if st.button("Submit"):
                if query_text:
                    response = query_data(filtered_df, query_text, pre_prompt)
                    st.markdown(response)
                else:
                    st.warning("Please enter a question or select a predefined question.")
        else:
            st.info("AI-assisted analysis is disabled. Please provide an API key to enable this feature.")

        # Download Data as CSV
        if st.button("Download Data as CSV"):
            output = download_csv(filtered_df, "grants_data_chart.csv")
            st.markdown(output, unsafe_allow_html=True)

        st.write("""
        We hope you find the Grant Amount Scatter Plot helpful in exploring the distribution of grant amounts over time. If you have any questions or suggestions, please don't hesitate to reach out.

        Happy exploring!
        """)
