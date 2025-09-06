import plotly.express as px
import streamlit as st
import pandas as pd

from utils.utils import download_excel, download_csv, generate_page_prompt
from utils.chat_panel import chat_panel


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
    # Guard clause
    if selected_chart != "Grant Amount Distribution":
        return

    # Display the header and description
    st.header("Grant Amount Distribution w AI Chat")
    st.write("""
        Dive into the dynamic landscape of grant funding with our interactive distribution chart. This tool lets you visualize how grants are dispersed across various USD clusters, offering a clear view of funding trends and concentrations. Select different clusters to tailor the data shown and discover patterns at a glance.
        """)

    # Validate required columns
    req_cols = {"amount_usd_cluster", "amount_usd"}
    if missing := sorted(req_cols - set(grouped_df.columns)):
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Settings & filters
    with st.expander("Distribution Settings", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric = st.selectbox("Aggregate by", ["Total Amount", "Count"], index=0)
        with col2:
            top_n = st.slider("Top N clusters", min_value=3, max_value=30, value=15)
        with col3:
            log_y = st.toggle("Log scale (Y)", value=False)
        with col4:
            sort_dir = st.selectbox("Sort", ["Descending", "Ascending"], index=0)

    # Display visualizations
    cluster_options = grouped_df['amount_usd_cluster'].dropna().astype(str).unique().tolist()
    cluster_options.sort()
    if not cluster_options:
        st.info("No cluster data available.")
        return
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options)

    # Work on a copy to avoid chained assignment issues
    grouped_df = grouped_df.copy()
    grouped_df['amount_usd_cluster'] = grouped_df['amount_usd_cluster'].astype(str)

    @st.cache_data(show_spinner=False)
    def _filter_clusters(gdf: pd.DataFrame, clusters: tuple[str, ...]) -> pd.DataFrame:
        return gdf[gdf['amount_usd_cluster'].isin(list(clusters))]

    filtered_df = _filter_clusters(grouped_df, tuple(selected_clusters)) if selected_clusters else grouped_df.iloc[0:0]

    if filtered_df.empty:
        st.info("No data for the selected clusters.")
        return

    # Aggregate (cached)
    @st.cache_data(show_spinner=False)
    def _aggregate_distribution(fdf: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
        if metric == "Total Amount":
            agg_df = (
                fdf.groupby('amount_usd_cluster', as_index=False)['amount_usd'].sum()
                .rename(columns={"amount_usd": "value"})  # type: ignore[call-arg]
            )
            label = "Total Amount (USD)"
        else:
            agg_df = (
                fdf.groupby('amount_usd_cluster', as_index=False)['grant_key'].count()
                .rename(columns={"grant_key": "value"})  # type: ignore[call-arg]
            )
            label = "Grant Count"
        return agg_df, label

    agg, y_label = _aggregate_distribution(filtered_df, metric)

    agg = agg.sort_values('value', ascending=(sort_dir == "Ascending")).head(int(top_n))

    fig = px.bar(
        agg,
        x='amount_usd_cluster',
        y='value',
        color='amount_usd_cluster',
        title=f"Grant Distribution by USD Cluster ({'Amount' if metric=='Total Amount' else 'Count'})",
    )
    fig.update_layout(xaxis_title='USD Cluster', yaxis_title=y_label, showlegend=False)
    if log_y:
        fig.update_yaxes(type='log')
    st.plotly_chart(fig, use_container_width=True)

    if ai_enabled:
        additional_context = "the distribution of grant amounts across different USD clusters"
        pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)
        chat_panel(filtered_df, pre_prompt, state_key="distribution_clusters", title="Distribution — AI Assistant")
    else:
        st.info("AI-assisted analysis is disabled. Provide an API key to enable this feature.")

    # Downloads
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        if st.button("Download Aggregated Data (Excel)"):
            download_excel(agg, "grants_distribution_agg.xlsx")
    with dcol2:
        if st.button("Download Filtered Rows (CSV)"):
            link = download_csv(filtered_df, "grants_distribution_filtered.csv")
            st.markdown(link, unsafe_allow_html=True)

    st.markdown(""" This app was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using the latest methods for enabling AI to Chat with Data. It also uses the Candid API, Streamlit, Plotly, and other open-source libraries. Generative AI solutions such as OpenAI GPT-4 and Claude Opus were used to generate portions of the source code.
                    """)
