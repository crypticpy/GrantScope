import streamlit as st
import plotly.express as px

def general_analysis_relationships(df, grouped_df, selected_chart, selected_role):
    if selected_chart == "General Analysis of Relationships":
        st.header("General Analysis of Relationships")
        st.write("Explore the relationships between various factors and the award amount.")

        unique_grants_df = df.drop_duplicates(subset=['grant_key'])

        st.subheader("Relationship between Grant Description Length and Award Amount")
        unique_grants_df['description_word_count'] = unique_grants_df['grant_description'].apply(
            lambda x: len(str(x).split()))
        fig = px.scatter(unique_grants_df, x='description_word_count', y='amount_usd', opacity=0.5,
                         title="Grant Description Length vs. Award Amount")
        fig.update_layout(xaxis_title='Number of Words in Grant Description', yaxis_title='Award Amount (USD)',
                          width=800, height=600)
        st.plotly_chart(fig)

        st.subheader("Distribution of Award Amounts by Different Factors")
        factors = ['grant_strategy_tran', 'grant_population_tran', 'grant_geo_area_tran', 'funder_name']
        selected_factor = st.selectbox("Select Factor", options=factors)

        exploded_df = unique_grants_df.assign(
            **{selected_factor: unique_grants_df[selected_factor].str.split(';')}).explode(selected_factor)

        fig = px.box(exploded_df, x=selected_factor, y='amount_usd',
                     title=f"Award Amount Distribution by {selected_factor}")
        fig.update_layout(xaxis_title=selected_factor, yaxis_title='Award Amount (USD)',
                          width=800, height=600, boxmode='group')
        st.plotly_chart(fig)

        st.subheader("Average Award Amount by Different Factors")
        avg_amount_by_factor = exploded_df.groupby(selected_factor)['amount_usd'].mean().reset_index()
        avg_amount_by_factor = avg_amount_by_factor.sort_values('amount_usd', ascending=False)
        fig = px.bar(avg_amount_by_factor, x=selected_factor, y='amount_usd',
                     title=f"Average Award Amount by {selected_factor}")
        fig.update_layout(xaxis_title=selected_factor, yaxis_title='Average Award Amount (USD)',
                          width=800, height=600, xaxis_tickangle=-45, xaxis_tickfont=dict(size=10))
        st.plotly_chart(fig)

        st.subheader("Funder Affinity Analysis")
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
        fig.update_layout(xaxis_title=selected_affinity_factor, yaxis_title='Total Award Amount (USD)',
                          width=800, height=600, xaxis_tickangle=-45, xaxis_tickfont=dict(size=10))

        st.plotly_chart(fig)

        if st.checkbox("Show Underlying Data"):
            st.write(unique_grants_df)

        if st.button("Download Data as CSV"):
            csv = unique_grants_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="grant_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)