import streamlit as st
import plotly.express as px
from utils import download_csv


def general_analysis_relationships(df, grouped_df, selected_chart, selected_role):
    if selected_chart == "General Analysis of Relationships":
        st.header("General Analysis of Relationships")
        st.write("""
        Welcome to the General Analysis of Relationships page! This section of the GrantScope application is designed to help you uncover meaningful connections and trends within the grant data. By exploring the relationships between various factors and the award amount, you can gain valuable insights to inform your grant-related decisions.

        The interactive visualizations on this page allow you to examine how different aspects of grants, such as the length of the grant description, funding strategies, target populations, and geographical areas, correlate with the awarded amounts. You can also investigate the affinity of specific funders towards certain subjects, populations, or strategies.

        To get started, simply select the desired factors from the dropdown menus and the application will generate informative plots for you to analyze. Feel free to upload your own dataset to uncover insights tailored to your specific needs.
        """)

        unique_grants_df = df.drop_duplicates(subset=['grant_key'])

        st.subheader("Relationship between Grant Description Length and Award Amount")
        st.write("Explore how the number of words in a grant description correlates with the award amount.")
        unique_grants_df['description_word_count'] = unique_grants_df['grant_description'].apply(
            lambda x: len(str(x).split()))
        fig = px.scatter(unique_grants_df, x='description_word_count', y='amount_usd', opacity=0.5,
                         title="Grant Description Length vs. Award Amount")
        fig.update_layout(xaxis_title='Number of Words in Grant Description', yaxis_title='Award Amount (USD)',
                          width=800, height=600)
        st.plotly_chart(fig)

        st.subheader("Average Award Amount by Different Factors")
        st.write(
            "Investigate how award amounts are distributed across various factors. Choose your factos and explore a bar chart or box plot."
            " You can select factors such as grant strategy, grant population, grant geographical area, and funder name."
            " The box plot provides a visual representation of the distribution of award amounts within each category. "
            " This option also allows you to identify potential outliers and variations in award amounts.")

        factors = ['grant_strategy_tran', 'grant_population_tran', 'grant_geo_area_tran', 'funder_name']
        selected_factor = st.selectbox("Select Factor", options=factors)

        exploded_df = df.assign(**{selected_factor: df[selected_factor].str.split(';')}).explode(selected_factor)
        avg_amount_by_factor = exploded_df.groupby(selected_factor)['amount_usd'].mean().reset_index()
        avg_amount_by_factor = avg_amount_by_factor.sort_values('amount_usd', ascending=False)

        chart_type = st.radio("Select Chart Type", options=["Bar Chart", "Box Plot"])

        if chart_type == "Bar Chart":
            fig = px.bar(avg_amount_by_factor, x=selected_factor, y='amount_usd',
                         title=f"Average Award Amount by {selected_factor}")
            fig.update_layout(xaxis_title=selected_factor, yaxis_title='Average Award Amount (USD)', width=800,
                              height=600,
                              xaxis_tickangle=-45, xaxis_tickfont=dict(size=10))
        else:
            fig = px.box(exploded_df, x=selected_factor, y='amount_usd',
                         title=f"Award Amount Distribution by {selected_factor}")
            fig.update_layout(xaxis_title=selected_factor, yaxis_title='Award Amount (USD)', width=800, height=600,
                              boxmode='group')

        st.plotly_chart(fig)

        st.subheader("Funder Affinity Analysis")
        st.write("Analyze the affinity of a specific funder towards certain subjects, populations, or strategies.")
        funders = unique_grants_df['funder_name'].unique().tolist()
        selected_funder = st.selectbox("Select Funder", options=funders)
        affinity_factors = ['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran']
        selected_affinity_factor = st.selectbox("Select Affinity Factor", options=affinity_factors)
        funder_grants_df = unique_grants_df[unique_grants_df['funder_name'] == selected_funder]
        exploded_funder_df = funder_grants_df.assign(
            **{selected_affinity_factor: funder_grants_df[selected_affinity_factor].str.split(';')}).explode(
            selected_affinity_factor)
        funder_affinity = exploded_funder_df.groupby(selected_affinity_factor)['amount_usd'].sum().reset_index()
        funder_affinity = funder_affinity.sort_values('amount_usd', ascending=False)
        fig = px.bar(funder_affinity, x=selected_affinity_factor, y='amount_usd',
                     title=f"Funder Affinity: {selected_funder} - {selected_affinity_factor}")
        fig.update_layout(xaxis_title=selected_affinity_factor, yaxis_title='Total Award Amount (USD)', width=800,
                          height=600,
                          xaxis_tickangle=-45, xaxis_tickfont=dict(size=10))
        st.plotly_chart(fig)

        st.write("""
        We hope that this General Analysis of Relationships page helps you uncover valuable insights and trends within the grant data. If you have any questions or need further assistance, please don't hesitate to reach out.

        Happy exploring!
        """)

        if st.checkbox("Show Underlying Data"):
            st.write(unique_grants_df)

        if st.button("Download Data as CSV"):
            href = download_csv(unique_grants_df, "grant_data.csv")
            st.markdown(href, unsafe_allow_html=True)