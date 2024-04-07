import streamlit as st
import plotly.express as px

def grant_amount_distribution(df, grouped_df, selected_chart, selected_role):

    if selected_chart == "Grant Amount Distribution":
        st.header("Grant Amount Distribution")
        st.write(
            "Explore the distribution of grant amounts across different USD clusters and apply optional drill-down filters.")

        cluster_options = grouped_df['amount_usd_cluster'].unique().tolist()
        selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options)

        filtered_df = grouped_df[grouped_df['amount_usd_cluster'].isin(selected_clusters)]

        st.divider()
        st.write(
            "It is recommended to use the multi select options above to narrow your scope before applying these drill down filters. Otherwise the chart may take a long time to load, and there will be too many options to sort.")
        col1, col2 = st.columns(2)
        with col1:
            drill_down_options = st.expander("Optional Drill-Down Filters", expanded=False)
            with drill_down_options:
                subject_options = filtered_df['grant_subject_tran'].unique().tolist()
                selected_subjects = st.multiselect("Select Grant Subjects", options=subject_options,
                                                   default=subject_options,
                                                   key='subject_select')

                population_options = filtered_df['grant_population_tran'].unique().tolist()
                selected_populations = st.multiselect("Select Grant Populations", options=population_options,
                                                      default=population_options, key='population_select')

                strategy_options = filtered_df['grant_strategy_tran'].unique().tolist()
                selected_strategies = st.multiselect("Select Grant Strategies", options=strategy_options,
                                                     default=strategy_options, key='strategy_select')

            apply_filters = st.button("Apply Filters")

        with col2:
            chart_type = st.radio("Select Chart Type", options=["Bar Chart", "Treemap"], index=0)

        filtered_subjects = selected_subjects if apply_filters else subject_options
        filtered_populations = selected_populations if apply_filters else population_options
        filtered_strategies = selected_strategies if apply_filters else strategy_options

        filtered_df = filtered_df[
            (filtered_df['grant_subject_tran'].isin(filtered_subjects)) &
            (filtered_df['grant_population_tran'].isin(filtered_populations)) &
            (filtered_df['grant_strategy_tran'].isin(filtered_strategies))
            ]

        if chart_type == "Bar Chart":
            fig = px.bar(filtered_df, x='amount_usd_cluster', y='amount_usd', color='amount_usd_cluster',
                         title="Grant Amount Distribution by USD Cluster",
                         custom_data=['grant_key', 'grant_description'])
            fig.update_layout(xaxis_title='USD Cluster', yaxis_title='Total Grant Amount')
            fig.update_traces(
                hovertemplate='<b>USD Cluster:</b> %{x}<br><b>Total Grant Amount:</b> %{y}<br><br><b>Grant Key:</b> %{customdata[0]}<br><b>Grant Description:</b> %{customdata[1]}')
        else:
            fig = px.treemap(filtered_df, path=['amount_usd_cluster'], values='amount_usd',
                             title="Grant Amount Distribution by USD Cluster (Treemap)")

        st.plotly_chart(fig)

        expander = st.expander("Show Grant Descriptions for Selected Filters", expanded=False)
        with expander:
            st.write("### Grant Descriptions:")
            for _, row in filtered_df.iterrows():
                st.write(f"- **Grant Key:** {row['grant_key']}")
                st.write(f"  **Description:** {row['grant_description']}")
                st.write("---")

        if st.checkbox("Show Underlying Data for Chart"):
            st.write(filtered_df)

        if st.button("Download Data for Chart"):
            download_excel(filtered_df, "grants_data_chart.xlsx")