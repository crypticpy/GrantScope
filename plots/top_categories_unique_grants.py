import streamlit as st
import plotly.express as px
from io import BytesIO
from textwrap import shorten

def top_categories_unique_grants(df, grouped_df, selected_chart, selected_role):

    if selected_chart == "Top Categories by Unique Grant Count":

        st.header("Top Categories by Unique Grant Count")

        st.write("Explore the distribution of unique grant counts across different categorical variables.")

        key_categorical_columns = ['funder_type', 'recip_organization_tran', 'grant_subject_tran', 'grant_population_tran',
                                   'grant_strategy_tran', 'year_issued']

        col1, col2 = st.columns(2)

        with col1:

            selected_categorical = st.selectbox("Select Categorical Variable", options=key_categorical_columns)

            top_n = st.slider("Number of Top Categories", min_value=5, max_value=20, value=10, step=1)

        with col2:

            chart_type = st.selectbox("Select Chart Type", options=["Bar Chart", "Pie Chart", "Treemap"])

            sort_order = st.radio("Sort Order", options=["Descending", "Ascending"])

        normalized_counts = df.groupby(selected_categorical)['grant_key'].nunique().sort_values(
            ascending=(sort_order == "Ascending")).reset_index()

        normalized_counts.columns = [selected_categorical, 'Unique Grant Keys']

        normalized_counts['truncated_col'] = normalized_counts[selected_categorical].apply(
            lambda x: shorten(x, width=30, placeholder="..."))

        if chart_type == "Bar Chart":
            fig = px.bar(normalized_counts.head(top_n), x='Unique Grant Keys', y='truncated_col', orientation='h',
                         title=f"Top {top_n} Categories in {selected_categorical}", hover_data={selected_categorical: True})
            fig.update_layout(yaxis_title=selected_categorical, margin=dict(l=0, r=0, t=30, b=0))
        elif chart_type == "Pie Chart":
            fig = px.pie(normalized_counts.head(top_n), values='Unique Grant Keys', names='truncated_col',
                         title=f"Distribution of Unique Grant Keys Across Top {top_n} Categories in {selected_categorical}")
        else:  # Treemap
            fig = px.treemap(normalized_counts.head(top_n), path=['truncated_col'], values='Unique Grant Keys',
                             title=f"Treemap of Unique Grant Keys Across Top {top_n} Categories in {selected_categorical}")

        st.plotly_chart(fig)

        st.write(
            f"Top {top_n} Categories account for {normalized_counts.head(top_n)['Unique Grant Keys'].sum() / normalized_counts['Unique Grant Keys'].sum():.2%} of total unique grants")

        expander = st.expander(f"Show Grants for Selected {selected_categorical} Category")
        with expander:

            selected_category = st.selectbox(f"Select {selected_categorical} Category",
                                             options=normalized_counts[selected_categorical])

            category_grants = df[df[selected_categorical] == selected_category].drop_duplicates(subset=['grant_key'])

            if not category_grants.empty:
                st.write(f"### Grant Details for {selected_category}:")
                grant_details = category_grants[
                    ['grant_key', 'grant_description', 'amount_usd', 'recip_organization_tran', 'year_issued']]
                st.dataframe(grant_details)
            else:

                st.write(f"No grants found for the selected category: {selected_category}")

        if st.button("Download Data for Chart"):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            normalized_counts.to_excel(writer, index=False, sheet_name='Top Categories')
            category_grants.to_excel(writer, index=False, sheet_name='Grants for Selected Category')
            writer.close()
            output.seek(0)
            b64 = base64.b64encode(output.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_chart.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)


        if st.button("Clear Cache"):
                clear_cache()
                st.session_state.cache_initialized = False
                st.rerun()