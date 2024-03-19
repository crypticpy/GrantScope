import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
from wordcloud import WordCloud, STOPWORDS
from textwrap import shorten
import base64
from io import BytesIO
import json
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class Grant:
    funder_key: str
    funder_profile_url: str
    funder_name: str
    funder_city: str
    funder_state: str
    funder_country: str
    funder_type: str
    funder_zipcode: str
    funder_country_code: str
    funder_ein: str
    funder_gs_profile_update_level: str
    recip_key: str
    recip_name: str
    recip_city: str
    recip_state: str
    recip_country: str
    recip_zipcode: str
    recip_country_code: str
    recip_ein: str
    recip_organization_code: str
    recip_organization_tran: str
    recip_gs_profile_link: str
    recip_gs_profile_update_level: str
    grant_key: str
    amount_usd: int
    grant_subject_code: str
    grant_subject_tran: str
    grant_population_code: str
    grant_population_tran: str
    grant_strategy_code: str
    grant_strategy_tran: str
    grant_transaction_code: str
    grant_transaction_tran: str
    grant_geo_area_code: str
    grant_geo_area_tran: str
    year_issued: str
    grant_duration: str
    grant_description: str
    last_updated: str

@dataclass
class Grants:
    grants: List[Grant]

@st.cache_data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    grants = Grants(grants=[Grant(**grant) for grant in data['grants']])
    return grants

def preprocess_data(grants):
    df = pd.DataFrame([asdict(grant) for grant in grants.grants])

    df['grant_index'] = df['grant_key']

    code_columns = [col for col in df.columns if "_code" in col]
    tran_columns = [col for col in df.columns if "_tran" in col]

    for code_col, tran_col in zip(code_columns, tran_columns):
        df[code_col] = df[code_col].apply(lambda x: x.split(';') if isinstance(x, str) else ['Unknown'])
        df[tran_col] = df[tran_col].apply(lambda x: x.split(';') if isinstance(x, str) else ['Unknown'])
        df = df.explode(code_col)
        df = df.explode(tran_col)

    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    for col in text_columns:
        df[col] = df[col].fillna('Unknown')

    for col in numeric_columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    bins = [0, 50000, 100000, 500000, 1000000, np.inf]
    names = ['0-50k', '50k-100k', '100k-500k', '500k-1M', '1M+']

    df['amount_usd_cluster'] = pd.cut(df['amount_usd'], bins, labels=names)

    grouped_df = df.groupby('grant_index').first()

    return df, grouped_df

# Load data
file_path = 'fixed_ovp.json'
grants = load_data(file_path)
df, grouped_df = preprocess_data(grants)


# Define chart options for each user role
chart_options = {
    "Grant Analyst/Writer": [
        "Data Summary",
        "Grant Amount Distribution by USD Cluster",
        "Grant Amount vs Year Scatter Plot",
        "Grant Amount by Population and Strategy Heatmap",
        "Top Grant Description Words by USD Cluster",
        "Treemaps of Grant Amount by Subject, Population and Strategy",
        "Univariate Analysis of Numeric Columns",
        "Top Categories by Unique Grant Count"
    ],
    "Normal Grant User": [
        "Data Summary",
        "Grant Amount Distribution by USD Cluster",
        "Grant Amount vs Year Scatter Plot",
        "Grant Amount by Population and Strategy Heatmap",
        "Top Grant Description Words by USD Cluster",
        "Treemaps of Grant Amount by Subject, Population and Strategy"
    ]
}

# Sidebar menu
user_roles = ["Grant Analyst/Writer", "Normal Grant User"]
selected_role = st.sidebar.selectbox("Select User Role", options=user_roles)
selected_chart = st.sidebar.selectbox("Select Chart", options=chart_options[selected_role])

# Main content
st.title("Grant Analysis Dashboard")

if selected_chart == "Data Summary":
    st.header("1. Data Summary")
    # 1. Data Loading and Preprocessing
    st.write(df.head())
    unique_grant_keys = df['grant_key'].nunique()
    st.write(f"Total Unique Grants: {unique_grant_keys}")
    st.write(f"Total Unique Funders: {df['funder_name'].nunique()}")
    st.write(f"Total Unique Recipients: {df['recip_name'].nunique()}")

    st.header("Top Funders by Total Grant Amount")
    top_n = st.slider("Select the number of top funders to display", min_value=5, max_value=20, value=10, step=1)

    top_funders = df.groupby('funder_name')['amount_usd'].sum().nlargest(top_n).reset_index()

    fig = px.bar(top_funders, x='funder_name', y='amount_usd', title=f"Top {top_n} Funders by Total Grant Amount")
    fig.update_layout(xaxis_title='Funder Name', yaxis_title='Total Grant Amount (USD)')
    st.plotly_chart(fig)

    if st.checkbox("Show Data Table"):
        st.write(top_funders)

    st.header("Grant Distribution by Funder Type")

    funder_type_dist = df.groupby('funder_type')['amount_usd'].sum().reset_index()

    fig = px.pie(funder_type_dist, values='amount_usd', names='funder_type', title="Grant Distribution by Funder Type")
    st.plotly_chart(fig)


# Grant Amount Distribution by USD Cluster
elif selected_chart == "Grant Amount Distribution by USD Cluster":
    st.header("Grant Amount Distribution by USD Cluster")
    cluster_options = grouped_df['amount_usd_cluster'].unique().tolist()
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options)

    filtered_df = grouped_df[grouped_df['amount_usd_cluster'].isin(selected_clusters)]

    drill_down_options = st.expander("Optional Drill-Down Filters", expanded=False)
    with drill_down_options:
        subject_options = filtered_df['grant_subject_tran'].unique().tolist()
        selected_subjects = st.multiselect("Select Grant Subjects", options=subject_options, default=subject_options,
                                           key='subject_select')
        subject_submit = st.button("Apply Grant Subject Filters")

        population_options = filtered_df['grant_population_tran'].unique().tolist()
        selected_populations = st.multiselect("Select Grant Populations", options=population_options,
                                              default=population_options, key='population_select')
        population_submit = st.button("Apply Grant Population Filters")

        strategy_options = filtered_df['grant_strategy_tran'].unique().tolist()
        selected_strategies = st.multiselect("Select Grant Strategies", options=strategy_options,
                                             default=strategy_options, key='strategy_select')
        strategy_submit = st.button("Apply Grant Strategy Filters")

    filtered_subjects = selected_subjects if subject_submit else subject_options
    filtered_populations = selected_populations if population_submit else population_options
    filtered_strategies = selected_strategies if strategy_submit else strategy_options

    filtered_df = filtered_df[
        (filtered_df['grant_subject_tran'].isin(filtered_subjects)) &
        (filtered_df['grant_population_tran'].isin(filtered_populations)) &
        (filtered_df['grant_strategy_tran'].isin(filtered_strategies))
        ]

    fig = px.bar(filtered_df, x='amount_usd_cluster', y='amount_usd', color='amount_usd_cluster')

    # Add click event to show underlying data
    fig.update_layout(clickmode='event+select')
    fig.update_traces(selector=dict(type='bar'), marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)

    st.plotly_chart(fig)

    if st.checkbox("Show Underlying Data for Chart"):
        st.write(filtered_df)

    if st.button("Download Data for Chart"):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        filtered_df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_chart.xlsx">Download Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)


# Grant Amount vs Year Scatter Plot
elif selected_chart == "Grant Amount vs Year Scatter Plot":
    st.header("Grant Amount vs Year Scatter Plot")

    # Get the unique years from the data
    unique_years = sorted(df['year_issued'].unique())

    if len(unique_years) == 1:
        # If there is only one year, display the year
        unique_year = int(unique_years[0])
        st.write(f"Data available for year: {unique_year}")
        start_year, end_year = unique_year, unique_year
    else:
        # If there are multiple years, display the year range slider
        start_year, end_year = st.slider(
            "Select Year Range",
            min_value=int(min(unique_years)),
            max_value=int(max(unique_years)),
            value=(int(min(unique_years)), int(max(unique_years)))
        )

    # Filter dataframe based on the selected year range
    filtered_df = grouped_df[
        (grouped_df['year_issued'].astype(int) >= start_year) &
        (grouped_df['year_issued'].astype(int) <= end_year)
    ]

    # Multi-select for choosing USD clusters
    cluster_options = filtered_df['amount_usd_cluster'].unique().tolist()
    selected_clusters = st.multiselect(
        "Select USD Clusters",
        options=cluster_options,
        default=cluster_options,
        key='scatter_clusters'
    )

    # Filter the dataframe based on the selected clusters
    filtered_df = filtered_df[filtered_df['amount_usd_cluster'].isin(selected_clusters)]

    # Create the scatter plot
    fig = px.scatter(
        filtered_df,
        x='year_issued',
        y='amount_usd',
        color='amount_usd_cluster',
        hover_data=['grant_key', 'grant_description']
    )

    # Update the layout of the figure
    fig.update_layout(
        title='Grant Amount by Year' if len(unique_years) > 1 else f'Grant Amount for Year {start_year}',
        xaxis_title='Year Issued',
        yaxis_title='Amount (USD)',
        legend_title_text='USD Cluster',
        legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"),
        clickmode='event+select'
    )

    # Update x-axis to display years as integers
    fig.update_xaxes(tickformat='d')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Checkbox to allow the user to choose to view the underlying data
    if st.checkbox("Show Underlying Data for Chart"):
        st.write(filtered_df)

    # Button to download the data as an Excel file
    if st.button("Download Data for Chart"):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        filtered_df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_chart.xlsx">Download Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)

# Grant Amount by Population and Strategy Heatmap
elif selected_chart == "Grant Amount by Population and Strategy Heatmap":
    st.header("Grant Amount by Population and Strategy Heatmap")

    # Select the dimensions for the heatmap
    dimension1 = st.selectbox("Select Dimension 1", options=['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran'])
    dimension2 = st.selectbox("Select Dimension 2", options=['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran'], index=1)

    # Select the values for each dimension
    selected_values1 = st.multiselect(f"Select {dimension1.split('_')[1].capitalize()}s", options=grouped_df[dimension1].unique(), default=grouped_df[dimension1].unique())
    selected_values2 = st.multiselect(f"Select {dimension2.split('_')[1].capitalize()}s", options=grouped_df[dimension2].unique(), default=grouped_df[dimension2].unique())

    # Create the pivot table for the heatmap
    pivot_table = grouped_df[
        grouped_df[dimension1].isin(selected_values1) &
        grouped_df[dimension2].isin(selected_values2)
    ].groupby([dimension1, dimension2])['amount_usd'].sum().unstack().fillna(0)

    # Create the Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        x=pivot_table.columns,
        y=pivot_table.index,
        z=pivot_table.values,
        colorscale='Plasma'
    ))

    fig.update_layout(
        title=f'Total Grant Amount by {dimension1.split("_")[1].capitalize()} and {dimension2.split("_")[1].capitalize()}',
        xaxis_title=dimension2.split('_')[1].capitalize(),
        yaxis_title=dimension1.split('_')[1].capitalize(),
        width=800,
        height=800
    )

    # Display the Plotly heatmap
    st.plotly_chart(fig)

    # Download button for the data
    if st.button("Download Data for Chart"):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        grouped_df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_chart.xlsx">Download Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Dive into specific cell data
    if st.checkbox("Dive Grant Cross Section Data"):
        selected_value1 = st.selectbox(f"Select {dimension1.split('_')[1].capitalize()}", options=selected_values1)

        # Dynamically update the values for dimension2 based on the selected value for dimension1
        filtered_df = grouped_df[grouped_df[dimension1] == selected_value1]
        available_values2 = filtered_df[dimension2].unique().tolist()

        if available_values2:
            selected_value2 = st.selectbox(f"Select {dimension2.split('_')[1].capitalize()}", options=available_values2)
            cell_grants = grouped_df[
                (grouped_df[dimension1] == selected_value1) &
                (grouped_df[dimension2] == selected_value2)
            ]
            st.write(cell_grants)
        else:
            st.write(f"No {dimension2.split('_')[1]}s available for the selected {dimension1.split('_')[1]}.")

# Top Grant Description Words by USD Cluster
elif selected_chart == "Top Grant Description Words by USD Cluster":
    st.header("Top Grant Description Words by USD Cluster")
    cluster_options = grouped_df['amount_usd_cluster'].unique().tolist()
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options,
                                       key='wordcloud_clusters')

    stopwords = set(STOPWORDS)
    additional_stopwords = {'public', 'Public', 'health', 'Health', 'and', 'And', 'to', 'To', 'of', 'Of', 'the', 'The',
                            'a', 'A', 'by', 'By', 'in', 'In', 'for', 'For', 'with', 'With', 'on', 'On', 'is', 'Is',
                            'that', 'That', 'are', 'Are', 'as', 'As', 'be', 'Be', 'this', 'This', 'will', 'Will', 'at',
                            'At', 'from', 'From', 'or', 'Or', 'an', 'An', 'which', 'Which', 'have', 'Have', 'it', 'It',
                            'not', 'Not', 'who', 'Who', 'their', 'Their', 'we', 'We', 'support', 'Support', 'project',
                            'Project'}
    stopwords.update(additional_stopwords)

    for cluster in selected_clusters:
        top_grants = grouped_df[grouped_df['amount_usd_cluster'] == cluster].nlargest(20, 'amount_usd')
        text = ' '.join(top_grants['grant_description'])

        wordcloud = WordCloud(stopwords=stopwords, width=800, height=400).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud of Grant Descriptions for {cluster} Cluster')
        st.pyplot(fig)

        words = [word for word in text.split() if word.lower() not in stopwords]
        word_freq = pd.Series(words).value_counts()
        st.write(f"Top Words for {cluster} Cluster:")
        st.write(word_freq.head(20))

# Treemaps of Grant Amount by Subject, Population and Strategy
elif selected_chart == "Treemaps of Grant Amount by Subject, Population and Strategy":
    st.header("Treemaps of Grant Amount by Subject, Population and Strategy")
    analyze_column = st.radio("Select Variable for Treemap",
                              options=['grant_strategy_tran', 'grant_subject_tran', 'grant_population_tran'])

    for label in grouped_df['amount_usd_cluster'].unique():
        filtered_data = grouped_df[grouped_df['amount_usd_cluster'] == label]
        grouped_data = filtered_data.groupby(analyze_column)['amount_usd'].sum().reset_index().sort_values(by='amount_usd',
                                                                                                           ascending=False)

        fig = px.treemap(grouped_data, path=[analyze_column], values='amount_usd',
                         title=f"Treemap: Sum of Amount in USD by {analyze_column} for {label} USD range")
        st.plotly_chart(fig)

        if st.checkbox(f"Show Grants for Selected {analyze_column} Block", key=f"{label}_{analyze_column}"):
            selected_block = st.selectbox(f"Select {analyze_column} Block", options=grouped_data[analyze_column])
            block_grants = filtered_data[filtered_data[analyze_column] == selected_block]
            st.write(block_grants)

        if st.button(f"Download Data for {label} USD range", key=f"download_{label}"):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            filtered_data.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
            output.seek(0)
            b64 = base64.b64encode(output.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_{label}.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)

    # Add summary table of all grants
    st.subheader("All Grants Summary")
    grant_summary = grouped_df[
        ['grant_key', 'amount_usd', 'year_issued', 'grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran']]
    st.write(grant_summary)

# Univariate Analysis of Numeric Columns
elif selected_chart == "Univariate Analysis of Numeric Columns":
    st.header("Univariate Analysis of Numeric Columns")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if 'amount_usd_cluster' in numeric_columns:
        numeric_columns.remove('amount_usd_cluster')

    selected_numeric = st.radio("Select Numeric Variable", options=numeric_columns)

    fig = px.histogram(df, x=selected_numeric, nbins=50, title=f"Distribution of {selected_numeric}")
    st.plotly_chart(fig)

    fig = px.box(df, y=selected_numeric, title=f"Boxplot of {selected_numeric}")
    st.plotly_chart(fig)

    st.write(f"Summary Statistics for {selected_numeric}:")
    st.write(df[selected_numeric].describe())

# Top Categories by Unique Grant Count
elif selected_chart == "Top Categories by Unique Grant Count":
    st.header("Top Categories by Unique Grant Count")
    key_categorical_columns = ['funder_type', 'recip_organization_code', 'grant_subject_code', 'grant_population_code',
                               'grant_strategy_code', 'year_issued']
    selected_categorical = st.selectbox("Select Categorical Variable", options=key_categorical_columns)

    normalized_counts = df.groupby(selected_categorical)['grant_key'].nunique().sort_values(ascending=False).reset_index()
    normalized_counts.columns = [selected_categorical, 'Unique Grant Keys']
    normalized_counts['truncated_col'] = normalized_counts[selected_categorical].apply(
        lambda x: shorten(x, width=30, placeholder="..."))

    fig = px.bar(normalized_counts.head(10), x='Unique Grant Keys', y='truncated_col', orientation='h',
                 title=f"Top 10 Categories in {selected_categorical}")
    fig.update_layout(yaxis_title=selected_categorical)
    st.plotly_chart(fig)

    st.write(
        f"Top 10 Categories account for {normalized_counts.head(10)['Unique Grant Keys'].sum() / normalized_counts['Unique Grant Keys'].sum():.2%} of total unique grants")

    if st.checkbox(f"Show Grants for Selected {selected_categorical} Category"):
        selected_category = st.selectbox(f"Select {selected_categorical} Category",
                                         options=normalized_counts['truncated_col'])
        category_grants = df[df[selected_categorical] == selected_category]
        st.write(category_grants)

    if st.button("Download Data for Chart"):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_chart.xlsx">Download Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)