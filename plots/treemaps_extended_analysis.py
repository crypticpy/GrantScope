import plotly.express as px
import streamlit as st

from utils.utils import download_excel


def treemaps_extended_analysis(df, grouped_df, selected_chart, selected_role, ai_enabled):
    if selected_chart == "Treemaps with Extended Analysis":
        usd_range_options = ['All'] + sorted(grouped_df['amount_usd_cluster'].unique())

        st.header("Treemaps by Subject, Population and Strategy")
        st.write("""
        Welcome to the Treemaps with Extended Analysis page! This interactive visualization allows you to explore the distribution of grant amounts across different subjects, populations, and strategies using dynamic treemaps.

        To get started, select a variable for the treemap using the radio buttons below. You can choose to analyze the data by grant strategy, subject, or population. Then, use the dropdown menu to filter the data by a specific USD range or select 'All' to include all grants.

        Click on a box in the treemap to view more details about the selected category. The treemap will highlight the selected category, and additional insights will be displayed below the treemap.

        Feel free to download the data for the selected USD range as an Excel file for further analysis.
        """)

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
            by='amount_usd', ascending=True)

        fig = px.treemap(grouped_data, path=[analyze_column], values='amount_usd',
                         title=f"Treemap: Sum of Amount in USD by {analyze_column} for {selected_label} USD range",
                         hover_data={'amount_usd': ':.2f'})
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

        selected_block = None

        def update_selected_block(trace, points, selector):
            nonlocal selected_block
            if points.point_inds:
                selected_block = points.path[0][points.point_inds[0]]
            else:
                selected_block = None

        fig.data[0].on_click(update_selected_block)

        st.plotly_chart(fig)

        if selected_block:
            block_grants = filtered_data[filtered_data[analyze_column] == selected_block]

            if not block_grants.empty:
                st.subheader(f"Insights for {selected_block}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(label="Total Grants", value=len(block_grants))

                with col2:
                    st.metric(label="Total USD", value=f"${block_grants['amount_usd'].sum():,.2f}")

                with col3:
                    st.metric(label="Average Grant USD", value=f"${block_grants['amount_usd'].mean():,.2f}")

                top_funders = block_grants.groupby('funder_name')['amount_usd'].sum().nlargest(5).reset_index()
                top_funders['amount_usd'] = top_funders['amount_usd'].apply(lambda x: f"${x:,.2f}")

                top_recipients = block_grants.groupby('recip_name')['amount_usd'].sum().nlargest(5).reset_index()
                top_recipients['amount_usd'] = top_recipients['amount_usd'].apply(lambda x: f"${x:,.2f}")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top Funders")
                    st.table(top_funders)

                with col2:
                    st.subheader("Top Recipient Organizations")
                    st.table(top_recipients)

                st.subheader("Selected Grant Descriptions")
                block_grants_sorted = block_grants.sort_values(by='amount_usd', ascending=False)
                block_grants_sorted = block_grants_sorted[['grant_description', 'amount_usd']]
                st.dataframe(block_grants_sorted)

                if st.button(f"Download Data for {selected_block} Category"):
                    output = download_excel(block_grants, f"grants_data_{selected_block}.xlsx")
                    st.markdown(output, unsafe_allow_html=True)

        if st.button(f"Download Data for {selected_label} USD Range"):
            output = download_excel(filtered_data, f"grants_data_{selected_label}.xlsx")
            st.markdown(output, unsafe_allow_html=True)

        st.write("""
        We hope you find the Treemaps with Extended Analysis page useful in exploring the distribution of grant amounts and gaining insights into specific categories. If you have any questions or suggestions, please don't hesitate to reach out.

        Happy exploring!
        """)
