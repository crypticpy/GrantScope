# sourcery skip: all
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.utils import download_csv, download_excel, generate_page_prompt
from utils.chat_panel import chat_panel


def grant_amount_heatmap(df, grouped_df, selected_chart, selected_role, ai_enabled):
    if selected_chart != "Grant Amount Heatmap":
        return
    st.header("Grant Amount Heatmap")
    st.write(
        """
        Welcome to the Grant Amount Heatmap page! This interactive visualization allows you to explore the intersection of grant dimensions and identify meaningful funding patterns.

        To get started, select two dimensions from the dropdown menus. The heatmap will display the total grant amount for each combination of the selected dimensions. You can further refine the heatmap by selecting specific values for each dimension using the expandable multiselect menus.

        Hover over the heatmap cells to view the total grant amount for each combination. Click on a cell to explore the underlying grant details, including the grant key, description, and amount.

        Feel free to download the heatmap data as an Excel file or the grant details as a CSV file for further analysis.
        """
    )

    # Validate required structure
    req_cols = {"amount_usd", "grant_subject_tran", "grant_population_tran", "grant_strategy_tran"}
    if missing := sorted(req_cols - set(grouped_df.columns)):
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Work on copies and ensure types
    grouped_df = grouped_df.copy()
    grouped_df["amount_usd"] = pd.to_numeric(grouped_df["amount_usd"], errors="coerce").fillna(0)
    for c in ["grant_subject_tran", "grant_population_tran", "grant_strategy_tran"]:
        grouped_df[c] = grouped_df[c].fillna("").astype(str)

    dimension_options = ["grant_subject_tran", "grant_population_tran", "grant_strategy_tran"]
    default_dim1 = dimension_options[0]

    col1, col2 = st.columns(2)
    with col1:
        dimension1 = st.selectbox(
            "Select Dimension 1", options=dimension_options, index=dimension_options.index(default_dim1)
        )
    with col2:
        dimension2 = st.selectbox(
            "Select Dimension 2", options=[d for d in dimension_options if d != dimension1], index=0
        )

    with st.expander("Heatmap Settings", expanded=False):
        colA, colB, _ = st.columns(3)
        with colA:
            colorscale = st.selectbox("Colorscale", ["Plasma", "Viridis", "Cividis", "Inferno", "Magma", "Blues", "Greens", "Reds"], index=0)
        with colB:
            normalize = st.toggle("Normalize by row", value=False, help="Show each row as proportion of its total")
    # Value labels are shown on hover; keeping UI minimal for now.

    st.caption("Select individual values for each dimension to filter the heatmap.")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f"Select {dimension1.split('_')[1].capitalize()}s", expanded=False):
            selected_values1 = st.multiselect(
                f"Select {dimension1.split('_')[1].capitalize()}s",
                options=grouped_df[dimension1].unique(),
                default=grouped_df[dimension1].unique(),
            )
    with col2:
        with st.expander(f"Select {dimension2.split('_')[1].capitalize()}s", expanded=False):
            selected_values2 = st.multiselect(
                f"Select {dimension2.split('_')[1].capitalize()}s",
                options=grouped_df[dimension2].unique(),
                default=grouped_df[dimension2].unique(),
            )

    filtered_df = grouped_df[
        grouped_df[dimension1].isin(selected_values1) & grouped_df[dimension2].isin(selected_values2)
    ]

    @st.cache_data(show_spinner=False)
    def _pivot_heatmap(fdf: pd.DataFrame, d1: str, d2: str) -> pd.DataFrame:
        return fdf.groupby([d1, d2])["amount_usd"].sum().unstack().fillna(0)

    pivot_table = _pivot_heatmap(filtered_df, dimension1, dimension2)
    row_sums = pivot_table.sum(axis=1).replace(0, np.nan)
    normalized = (pivot_table.T / row_sums).T.fillna(0)
    # Branchless selection to avoid linter's swap-if-else suggestion
    selected_values = np.where(bool(normalize), normalized.values, pivot_table.values)
    pivot_table = pd.DataFrame(selected_values, index=pivot_table.index, columns=pivot_table.columns)

    value_label = "Proportion" if normalize else "Value"
    hover_value_fmt = "%{z:.2f}" if normalize else "%{z:,.0f}"
    colorbar_title = "Row Proportion" if normalize else "Total Grant Amount"

    hover_prefix = "<b>%{yaxis.title.text}</b>: %{y}<br><b>%{xaxis.title.text}</b>: %{x}<br>"
    hover_suffix = f"<b>{value_label}</b>: {hover_value_fmt}"
    hover_template = hover_prefix + hover_suffix

    fig = go.Figure(
        data=go.Heatmap(
            x=pivot_table.columns,
            y=pivot_table.index,
            z=pivot_table.values,
            colorscale=colorscale,
            hovertemplate=hover_template,
            colorbar=dict(title=colorbar_title),
        )
    )

    fig.update_layout(
        title=(
            f"Total Grant Amount by {dimension1.split('_')[1].capitalize()} and {dimension2.split('_')[1].capitalize()}"
        ),
        xaxis_title=dimension2.split("_")[1].capitalize(),
        yaxis_title=dimension1.split("_")[1].capitalize(),
        width=800,
        height=800,
    )

    st.plotly_chart(fig, use_container_width=True)

    if ai_enabled:
        additional_context = (
            f"the intersection of {dimension1.split('_')[1]} and {dimension2.split('_')[1]} dimensions, "
            f"filtered by {dimension1.split('_')[1]}s ({', '.join(map(str, selected_values1))}) and "
            f"{dimension2.split('_')[1]}s ({', '.join(map(str, selected_values2))})"
        )
        pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)
        chat_panel(filtered_df, pre_prompt, state_key="heatmap_chat", title="Heatmap — AI Assistant")
    else:
        st.info("AI-assisted analysis is disabled. Please provide an API key to enable this feature.")

    if st.button("Download Heatmap Data as Excel"):
        output = download_excel(pivot_table, "heatmap_data.xlsx")
        st.markdown(output, unsafe_allow_html=True)

    st.divider()

    st.subheader("Explore Grant Details Further")
    st.write(
        " Choose a grant subject and the matching populations will be displayed as options in the next selectbox."
        " This analysis will help you identify the value of common intersections between grant subjects and populations."
    )
    selected_value1 = st.selectbox(
        f"Select {dimension1.split('_')[1].capitalize()}", options=selected_values1
    )

    filtered_df2 = grouped_df[grouped_df[dimension1] == selected_value1]
    available_values2 = filtered_df2[dimension2].unique().tolist()

    if len(available_values2) > 0:
        selected_value2 = st.selectbox(
            f"Select {dimension2.split('_')[1].capitalize()}", options=available_values2
        )

        cell_grants = grouped_df[
            (grouped_df[dimension1] == selected_value1) & (grouped_df[dimension2] == selected_value2)
        ]

        if cell_grants.empty:
            st.write(
                f"No grants found for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}"
            )
        else:
            st.write(
                f"Grants for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}"
            )
            grant_details = cell_grants[["grant_key", "grant_description", "amount_usd"]]
            st.write(grant_details)

            if st.button("Download The Above Grant Details as CSV"):
                output = download_csv(grant_details, "grant_details.csv")
                st.markdown(output, unsafe_allow_html=True)
    else:
        st.write(
            f"No {dimension2.split('_')[1].capitalize()}s available for the selected {dimension1.split('_')[1].capitalize()}."
        )

    st.write(
        """
        We hope you find the Grant Amount Heatmap helpful in uncovering funding patterns and exploring grant details. If you have any questions or suggestions, please don't hesitate to reach out.

        Happy exploring!
        """
    )

    st.markdown(
        """ This app was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using the latest methods for enabling AI to Chat with Data. It also uses the Candid API, Streamlit, Plotly, and other open-source libraries. Generative AI solutions such as OpenAI GPT-4 and Claude Opus were used to generate portions of the source code.
                        """
    )