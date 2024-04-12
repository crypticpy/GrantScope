import streamlit as st
import plotly.express as px
import pandas as pd
from utils import download_csv


def grant_amount_scatter_plot(df, grouped_df, selected_chart, selected_role):
    if selected_chart == "Grant Amount Scatter Plot":
        st.header("Grant Amount Scatter Plot")
        st.write("""
        Welcome to the Grant Amount Scatter Plot page! This interactive visualization allows you to explore the distribution of grant amounts over time.
        
        This visualization makes it easy to spot trends and patterns in grant amounts across different USD clusters.

        The scatter plot dynamically updates based on the selected USD clusters and year range available in the data. Use the filters on the left to customize your view:

        1. Select the desired start and end years using the number input fields.
        2. Choose the USD clusters you want to include in the plot using the multiselect dropdown.
        3. Adjust the marker size and opacity using the sliders to enhance the visual representation.

        Hover over the data points to view details such as the grant key, description, and amount. You can also click on the legend items to toggle the visibility of specific USD clusters or double-click to isolate a single cluster.

        Feel free to download the underlying data as a CSV file for further analysis.
        """)

        unique_years = sorted(df['year_issued'].unique())
        if len(unique_years) == 1:
            unique_year = int(unique_years[0])
            st.write(f"Data available for year: {unique_year}")
            start_year, end_year = unique_year, unique_year
        else:
            start_year = st.number_input("Start Year", min_value=int(min(unique_years)),
                                         max_value=int(max(unique_years)), value=int(min(unique_years)))
            end_year = st.number_input("End Year", min_value=int(min(unique_years)), max_value=int(max(unique_years)),
                                       value=int(max(unique_years)))

        filtered_df = grouped_df[
            (grouped_df['year_issued'].astype(int) >= start_year) &
            (grouped_df['year_issued'].astype(int) <= end_year)
            ]
        filtered_df = filtered_df[filtered_df['amount_usd'].notna() & (filtered_df['amount_usd'] > 0)]

        cluster_options = filtered_df['amount_usd_cluster'].unique().tolist()
        selected_clusters = st.multiselect(
            "Select USD Clusters",
            options=cluster_options,
            default=cluster_options,
            key='scatter_clusters'
        )
        filtered_df = filtered_df[filtered_df['amount_usd_cluster'].isin(selected_clusters)]

        marker_size = st.slider("Marker Size", min_value=1, max_value=20, value=5)
        opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

        filtered_df['year_issued'] = pd.to_datetime(filtered_df['year_issued'], format='%Y')
        unique_years = filtered_df['year_issued'].dt.year.unique()
        unique_years.sort()

        fig = px.scatter(
            filtered_df,
            x='year_issued',
            y='amount_usd',
            color='amount_usd_cluster',
            hover_data=['grant_key', 'grant_description', 'amount_usd'],
            opacity=opacity
        )
        fig.update_traces(marker=dict(size=marker_size))
        fig.update_layout(
            title='Grant Amount by Year',
            xaxis_title='Year Issued',
            yaxis_title='Amount (USD)',
            legend_title_text='USD Cluster',
            legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"),
            clickmode='event+select'
        )
        fig.update_xaxes(
            tickvals=pd.to_datetime(unique_years, format='%Y'),
            ticktext=unique_years
        )

        st.plotly_chart(fig)

        if st.checkbox("Show Underlying Data"):
            st.write(filtered_df)

        if st.button("Download Data as CSV"):
            output = download_csv(filtered_df, "grants_data_chart.csv")
            st.markdown(output, unsafe_allow_html=True)

        st.write("""
        We hope you find the Grant Amount Scatter Plot helpful in exploring the distribution of grant amounts over time. If you have any questions or suggestions, please don't hesitate to reach out.

        Happy exploring!
        """)