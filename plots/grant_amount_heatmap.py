import plotly.graph_objects as go
import streamlit as st

from loaders.llama_index_setup import query_data
from utils.utils import download_csv, download_excel, generate_page_prompt


def grant_amount_heatmap(df, grouped_df, selected_chart, selected_role, ai_enabled):
    if selected_chart == "Grant Amount Heatmap":
        st.header("Grant Amount Heatmap")
        st.write("""
        Welcome to the Grant Amount Heatmap page! This interactive visualization allows you to explore the intersection of grant dimensions and identify meaningful funding patterns.

        To get started, select two dimensions from the dropdown menus. The heatmap will display the total grant amount for each combination of the selected dimensions. You can further refine the heatmap by selecting specific values for each dimension using the expandable multiselect menus.

        Hover over the heatmap cells to view the total grant amount for each combination. Click on a cell to explore the underlying grant details, including the grant key, description, and amount.

        Feel free to download the heatmap data as an Excel file or the grant details as a CSV file for further analysis.
        """)

        dimension_options = ['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran']
        default_dim1, default_dim2 = dimension_options[:2]

        col1, col2 = st.columns(2)
        with col1:
            dimension1 = st.selectbox("Select Dimension 1", options=dimension_options,
                                      index=dimension_options.index(default_dim1))
        with col2:
            dimension2 = st.selectbox("Select Dimension 2", options=[d for d in dimension_options if d != dimension1],
                                      index=0)

        st.caption("Select individual values for each dimension to filter the heatmap.")

        col1, col2 = st.columns(2)
        with col1:
            with st.expander(f"Select {dimension1.split('_')[1].capitalize()}s", expanded=False):
                selected_values1 = st.multiselect(f"Select {dimension1.split('_')[1].capitalize()}s",
                                                  options=grouped_df[dimension1].unique(),
                                                  default=grouped_df[dimension1].unique())
        with col2:
            with st.expander(f"Select {dimension2.split('_')[1].capitalize()}s", expanded=False):
                selected_values2 = st.multiselect(f"Select {dimension2.split('_')[1].capitalize()}s",
                                                  options=grouped_df[dimension2].unique(),
                                                  default=grouped_df[dimension2].unique())

        filtered_df = grouped_df[
            grouped_df[dimension1].isin(selected_values1) &
            grouped_df[dimension2].isin(selected_values2)
            ]

        pivot_table = filtered_df.groupby([dimension1, dimension2])['amount_usd'].sum().unstack().fillna(0)

        fig = go.Figure(data=go.Heatmap(
            x=pivot_table.columns,
            y=pivot_table.index,
            z=pivot_table.values,
            colorscale='Plasma',
            hovertemplate='<b>%{yaxis.title.text}</b>: %{y}<br><b>%{xaxis.title.text}</b>: %{x}<br><b>Total Grant Amount</b>: %{z:,.0f}',
            colorbar=dict(title='Total Grant Amount')
        ))

        fig.update_layout(
            title=f'Total Grant Amount by {dimension1.split("_")[1].capitalize()} and {dimension2.split("_")[1].capitalize()}',
            xaxis_title=dimension2.split('_')[1].capitalize(),
            yaxis_title=dimension1.split('_')[1].capitalize(),
            width=800,
            height=800
        )

        st.plotly_chart(fig)

        if ai_enabled:
            # AI-Assisted Chat
            st.subheader("Heatmap Exploration with GPT-4 Assistant")
            st.write("Ask questions about the Grant Amount Heatmap to gain insights and explore the data further.")

            # Generate the custom prompt for the current page
            additional_context = f"the intersection of {dimension1.split('_')[1]} and {dimension2.split('_')[1]} dimensions, with data filtered by {dimension1.split('_')[1]}s ({', '.join(selected_values1)}) and {dimension2.split('_')[1]}s ({', '.join(selected_values2)})"
            pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

            # Predefined questions
            query_options = [
                f"What are the top 3 {dimension1.split('_')[1]}s by total grant amount for each {dimension2.split('_')[1]}?",
                f"Which {dimension2.split('_')[1]} has the highest total grant amount across all {dimension1.split('_')[1]}s?",
                f"Are there any notable patterns or correlations between {dimension1.split('_')[1]}s and {dimension2.split('_')[1]}s in terms of grant funding?",
                f"How does the distribution of grant amounts vary across different combinations of {dimension1.split('_')[1]}s and {dimension2.split('_')[1]}s?",
                f"What insights can we gain from the grant descriptions for the highest-funded combination of {dimension1.split('_')[1]} and {dimension2.split('_')[1]}?"
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

        if st.button("Download Heatmap Data as Excel"):
            output = download_excel(pivot_table, "heatmap_data.xlsx")
            st.markdown(output, unsafe_allow_html=True)

        st.divider()

        st.subheader("Explore Grant Details Further")
        st.write(
            " Choose a grant subject and the matching populations will be displayed as options in the next selectbox."
            " This analysis will help you identify the value of common intersections between grant subjects and populations.")
        selected_value1 = st.selectbox(f"Select {dimension1.split('_')[1].capitalize()}", options=selected_values1)

        filtered_df = grouped_df[grouped_df[dimension1] == selected_value1]
        available_values2 = filtered_df[dimension2].unique().tolist()

        if available_values2:
            selected_value2 = st.selectbox(f"Select {dimension2.split('_')[1].capitalize()}", options=available_values2)

            cell_grants = grouped_df[
                (grouped_df[dimension1] == selected_value1) &
                (grouped_df[dimension2] == selected_value2)
                ]

            if not cell_grants.empty:
                st.write(
                    f"Grants for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}")
                grant_details = cell_grants[['grant_key', 'grant_description', 'amount_usd']]
                st.write(grant_details)

                if st.button("Download The Above Grant Details as CSV"):
                    output = download_csv(grant_details, "grant_details.csv")
                    st.markdown(output, unsafe_allow_html=True)
            else:
                st.write(
                    f"No grants found for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}")
        else:
            st.write(
                f"No {dimension2.split('_')[1].capitalize()}s available for the selected {dimension1.split('_')[1].capitalize()}.")

        st.write("""
        We hope you find the Grant Amount Heatmap helpful in uncovering funding patterns and exploring grant details. If you have any questions or suggestions, please don't hesitate to reach out.

        Happy exploring!
        """)

        st.markdown(""" This app was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using the latest methods for enabling AI to Chat with Data. It also uses the Candid API, Streamlit, Plotly, and other open-source libraries. Generative AI solutions such as OpenAI GPT-4 and Claude Opus were used to generate portions of the source code.
                        """)