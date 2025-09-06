import pandas as pd
import plotly.express as px
import streamlit as st

from utils.chat_panel import chat_panel
from utils.utils import download_csv, generate_page_prompt


def grant_amount_scatter_plot(df, grouped_df, selected_chart, selected_role, ai_enabled):
    if selected_chart != "Grant Amount Scatter Plot":
        return

    st.header("Grant Amount Scatter Plot")
    st.write(
        """
        Welcome to the Grant Amount Scatter Plot page! This AI interactive visualization allows you to explore the distribution of grant amounts over time with support from GPT-4.

        This visualization makes it easy to spot trends and patterns in grant amounts across different USD clusters.

        The scatter plot dynamically updates based on the selected USD clusters and year range available in the data. Use the filters on the left to customize your view:

        1. Select the desired start and end years using the number input fields.
        2. Choose the USD clusters you want to include in the plot using the multiselect dropdown.
        3. Adjust the marker size and opacity using the sliders to enhance the visual representation.

        Hover over the data points to view details such as the grant key, description, and amount. You can also click on the legend items to toggle the visibility of specific USD clusters or double-click to isolate a single cluster.

        Feel free to download the underlying data as a CSV file for further analysis.
        """
    )

    # Validate required structure
    req_cols = {"year_issued", "amount_usd", "amount_usd_cluster", "grant_key", "grant_description"}
    if missing := sorted(req_cols - set(grouped_df.columns)):
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Ensure types and copy safety
    df = df.copy()
    grouped_df = grouped_df.copy()
    grouped_df["amount_usd"] = pd.to_numeric(grouped_df["amount_usd"], errors="coerce").fillna(0)
    grouped_df["year_issued"] = pd.to_numeric(grouped_df["year_issued"], errors="coerce").fillna(0).astype(int)
    grouped_df["amount_usd_cluster"] = grouped_df["amount_usd_cluster"].fillna("").astype(str)

    unique_years = sorted(grouped_df['year_issued'].unique())
    if len(unique_years) == 1:
        unique_year = int(unique_years[0])
        st.write(f"Data available for year: {unique_year}")
        start_year, end_year = unique_year, unique_year
    else:
        start_year = st.number_input(
            "Start Year",
            min_value=int(min(unique_years)),
            max_value=int(max(unique_years)),
            value=int(min(unique_years)),
        )
        end_year = st.number_input(
            "End Year",
            min_value=int(min(unique_years)),
            max_value=int(max(unique_years)),
            value=int(max(unique_years)),
        )

    @st.cache_data(show_spinner=False)
    def _filter_scatter(gdf: pd.DataFrame, s: int, e: int, clusters: tuple[str, ...]) -> pd.DataFrame:
        f = gdf[(gdf['year_issued'] >= s) & (gdf['year_issued'] <= e)]
        if clusters:
            f = f[f['amount_usd_cluster'].astype(str).isin(clusters)]
        return f[f['amount_usd'] > 0]

    cluster_options = grouped_df['amount_usd_cluster'].dropna().astype(str).unique().tolist()
    selected_clusters = st.multiselect(
        "Select USD Clusters",
        options=cluster_options,
        default=cluster_options,
        key='scatter_clusters',
    )
    filtered_df = _filter_scatter(grouped_df, int(start_year), int(end_year), tuple(map(str, selected_clusters)))

    if filtered_df.empty:
        st.warning("No data for the selected filters. Try expanding the year range or clusters.")
        return

    with st.expander("Scatter Settings", expanded=False):
        colA, colB, colC = st.columns(3)
        with colA:
            marker_size = st.slider("Marker Size", min_value=1, max_value=20, value=5)
        with colB:
            opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
        with colC:
            log_y = st.toggle("Log scale (Y)", value=False)

    filtered_df['year_issued'] = pd.to_datetime(filtered_df['year_issued'], format='%Y')
    years_for_ticks = sorted(filtered_df['year_issued'].dt.year.unique())

    fig = px.scatter(
        filtered_df,
        x='year_issued',
        y='amount_usd',
        color='amount_usd_cluster',
        hover_data=['grant_key', 'grant_description', 'amount_usd'],
        opacity=opacity,
    )
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(
        title='Grant Amount by Year',
        xaxis_title='Year Issued',
        yaxis_title='Amount (USD)',
        legend_title_text='USD Cluster',
        legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"),
        clickmode='event+select',
    )
    if years_for_ticks:
        fig.update_xaxes(
            tickvals=pd.to_datetime(years_for_ticks, format='%Y'),
            ticktext=years_for_ticks,
        )

    if log_y:
        fig.update_yaxes(type='log')
    st.plotly_chart(fig, use_container_width=True)

    if ai_enabled:
        additional_context = (
            f"the distribution of grant amounts over time, filtered by USD clusters ({', '.join(map(str, selected_clusters))}) "
            f"and year range ({start_year} to {end_year})"
        )
        pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)
        chat_panel(filtered_df, pre_prompt, state_key="scatter_chat", title="Scatter — AI Assistant")
    else:
        st.info("AI-assisted analysis is disabled. Please provide an API key to enable this feature.")

    # Download Data as CSV
    if st.button("Download Data as CSV"):
        output = download_csv(filtered_df, "grants_data_chart.csv")
        st.markdown(output, unsafe_allow_html=True)

    st.write(
        """
        We hope you find the Grant Amount Scatter Plot helpful in exploring the distribution of grant amounts over time. If you have any questions or suggestions, please don't hesitate to reach out.

        Happy exploring!
        """
    )
    st.markdown(
        """ This app was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using the latest methods for enabling AI to Chat with Data. It also uses the Candid API, Streamlit, Plotly, and other open-source libraries. Generative AI solutions such as OpenAI GPT-4 and Claude Opus were used to generate portions of the source code.
                        """
    )