import streamlit as st
import plotly.express as px
import pandas as pd
from io import BytesIO

def treemaps_extended_analysis(df, grouped_df, selected_chart, selected_role):
    if selected_chart == "Treemaps with Extended Analysis":

        usd_range_options = ['All'] + sorted(grouped_df['amount_usd_cluster'].unique())

        st.header("Treemaps by Subject, Population and Strategy")

        st.write(
            "Explore the distribution of grant amounts across different subjects, populations, and strategies using interactive treemaps.Select categories using drop downs below to investigate box details.")

        col1, col2 = st.columns(2)
        with col1:

            analyze_column = st.radio("Select Variable for Treemap",
                                      options=['grant_strategy_tran', 'grant_subject_tran', 'grant_population_tran'])
        with col2:

            selected_label = st.selectbox("Select USD Range", options=usd_range_options)

        if selected_label == 'All':
            filtered_data = grouped_df
        else:
            filtered_data = grouped_df[grouped_df['amount_usd_cluster'] == selected_label]

        grouped_data = filtered_data.groupby(analyze_column)['amount_usd'].sum().reset_index().sort_values(
            by='amount_usd',
            ascending=True)

        fig = px.treemap(grouped_data, path=[analyze_column], values='amount_usd',
                         title=f"Treemap: Sum of Amount in USD by {analyze_column} for {selected_label} USD range",
                         hover_data={'amount_usd': ':.2f'})

        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

        st.plotly_chart(fig)

        block_grants = pd.DataFrame()

        if not grouped_data.empty:
            selected_block = st.selectbox("Select Detail to View",
                                          options=[None] + sorted(grouped_data[analyze_column].unique()), index=0)
            if selected_block:
                block_grants = filtered_data[filtered_data[analyze_column] == selected_block]

                if not block_grants.empty:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Total Grants", value=len(block_grants))
                    with col2:
                        st.metric(label="Total USD", value=f"${block_grants['amount_usd'].sum():,.2f}")
                    with col3:
                        st.metric(label="Average Grant USD", value=f"${block_grants['amount_usd'].mean():,.2f}")

                    block_grants_sorted = block_grants.sort_values(by='amount_usd', ascending=False)
                    block_grants_sorted = block_grants_sorted[['grant_description', 'amount_usd']]
                    st.subheader("Selected Grant Descriptions")
                    st.dataframe(block_grants_sorted)

            if not block_grants.empty:
                st.subheader("Raw Data for Selected Detail")
                grant_summary = block_grants[
                    ['grant_key', 'amount_usd', 'year_issued', 'grant_subject_tran', 'grant_population_tran',
                     'grant_strategy_tran']]
                st.dataframe(grant_summary)
            else:
                st.write("No detailed data available for the selected options.")

        if st.button(f"Download Data for {selected_label} USD range"):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            filtered_data.to_excel(writer, index=False, sheet_name='Grants Data')
            writer.close()
            output.seek(0)
            b64 = base64.b64encode(output.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_{selected_label}.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)