import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from textwrap import shorten
import base64
from io import BytesIO
from streamlit_plotly_events import plotly_events


@st.cache_data
def load_data(file_path):
    df = pd.read_json(file_path)
    df = df['data']['rows']
    return df


def preprocess_data(df):
    # Normalize the json data into a dataframe
    df = pd.json_normalize(df)

    # Create a new column 'grant_index' that stores the original 'grant_key' for each row
    df['grant_index'] = df['grant_key']

    # Process semicolon-separated "_code" and "_tran" columns individually
    code_columns = [col for col in df.columns if "_code" in col]
    tran_columns = [col for col in df.columns if "_tran" in col]

    for code_col in code_columns:
        df[code_col] = df[code_col].str.split(';')
        df = df.explode(code_col)

    for tran_col in tran_columns:
        df[tran_col] = df[tran_col].str.split(';')
        df = df.explode(tran_col)

    # Fill missing values
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    for col in text_columns:
        df[col] = df[col].fillna('Unknown')

    for col in numeric_columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # Define the bins for the USD amount clusters
    bins = [0, 50000, 100000, 500000, 1000000, np.inf]
    names = ['0-50k', '50k-100k', '100k-500k', '500k-1M', '1M+']

    # Create a new column for the USD amount clusters
    df['amount_usd_cluster'] = pd.cut(df['amount_usd'], bins, labels=names)

    # Group by 'grant_index' to ensure we only count each grant once when summing 'amount_usd'
    grouped_df = df.groupby('grant_index').first()

    return df, grouped_df


# Load data
file_path = 'HITUpgradesFull_fix.txt'
df = load_data(file_path)
df, grouped_df = preprocess_data(df)

# Sidebar menu
chart_options = [
    "Data Summary",
    "Grant Amount Distribution by USD Cluster",
    "Grant Amount vs Year Scatter Plot",
    "Grant Amount by Population and Strategy Heatmap",
    "Top Grant Description Words by USD Cluster",
    "Treemaps of Grant Amount by Subject, Population and Strategy",
    "Univariate Analysis of Numeric Columns",
    "Top Categories by Unique Grant Count"
]
selected_chart = st.sidebar.selectbox("Select Chart", options=chart_options)

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

# 2. Grant Amount Distribution by USD Cluster
elif selected_chart == "Grant Amount Distribution by USD Cluster":
    st.header("2. Grant Amount Distribution by USD Cluster")
    cluster_options = grouped_df['amount_usd_cluster'].unique().tolist()
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options)
    filtered_df = grouped_df[grouped_df['amount_usd_cluster'].isin(selected_clusters)]
    fig = px.bar(filtered_df, x='amount_usd_cluster', y='amount_usd', color='amount_usd_cluster')

    # Add click event to show underlying data
    fig.update_layout(clickmode='event+select')
    fig.update_traces(selector=dict(type='bar'), marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)

    st.plotly_chart(fig)

    if st.checkbox("Show Underlying Data for Selected Cluster"):
        selected_points = plotly_events(fig, click_event=True, hover_event=False, override_width="100%")
        if selected_points:
            selected_cluster = selected_points[0]['x']
            cluster_grants = grouped_df[grouped_df['amount_usd_cluster'] == selected_cluster]
            st.write(cluster_grants)

# 3. Grant Amount vs Year Scatter Plot
elif selected_chart == "Grant Amount vs Year Scatter Plot":
    st.header("3. Grant Amount vs Year Scatter Plot")
    start_year, end_year = st.slider("Select Year Range", min_value=int(df['year_issued'].min()), max_value=int(df['year_issued'].max()), value=(int(df['year_issued'].min()), int(df['year_issued'].max())))
    filtered_df = grouped_df[(grouped_df['year_issued'] >= str(start_year)) & (grouped_df['year_issued'] <= str(end_year))]

    cluster_options = filtered_df['amount_usd_cluster'].unique().tolist()
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options, key='scatter_clusters')
    filtered_df = filtered_df[filtered_df['amount_usd_cluster'].isin(selected_clusters)]

    fig = px.scatter(filtered_df, x='year_issued', y='amount_usd', color='amount_usd_cluster', hover_data=['grant_key', 'grant_description'])
    fig.update_layout(title='Grant Amount by Year', xaxis_title='Year Issued', yaxis_title='Amount (USD)')

    # Add legend and cluster toggle
    fig.update_layout(legend_title_text='USD Cluster', legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"))

    st.plotly_chart(fig)

    # Add click event to show grant details
    fig.update_layout(clickmode='event+select')
    selected_points = plotly_events(fig, click_event=True, hover_event=False, override_width="100%")
    if selected_points:
        selected_grant = selected_points[0]['customdata'][0]
        grant_details = grouped_df[grouped_df['grant_key'] == selected_grant]
        st.write(grant_details)

# 4. Grant Amount by Population and Strategy Heatmap
elif selected_chart == "Grant Amount by Population and Strategy Heatmap":
    st.header("4. Grant Amount by Population and Strategy Heatmap")
    selected_populations = st.multiselect("Select Populations", options=grouped_df['grant_population_tran'].unique(),
                                          default=grouped_df['grant_population_tran'].unique())
    selected_strategies = st.multiselect("Select Strategies", options=grouped_df['grant_strategy_tran'].unique(),
                                         default=grouped_df['grant_strategy_tran'].unique())

    pivot_population_strategy = grouped_df[
        grouped_df['grant_population_tran'].isin(selected_populations) & grouped_df['grant_strategy_tran'].isin(
            selected_strategies)].groupby(['grant_population_tran', 'grant_strategy_tran'])[
        'amount_usd'].sum().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_population_strategy, cmap='plasma', ax=ax)
    ax.set_title('Total Grant Amount by Grant Population and Grant Strategy')
    st.pyplot(fig)

    if st.checkbox("Show Grants for Selected Cell"):
        selected_population = st.selectbox("Select Population", options=selected_populations)
        selected_strategy = st.selectbox("Select Strategy", options=selected_strategies)
        cell_grants = grouped_df[(grouped_df['grant_population_tran'] == selected_population) & (
                    grouped_df['grant_strategy_tran'] == selected_strategy)]
        st.write(cell_grants)

# 5. Top Grant Description Words by USD Cluster
elif selected_chart == "Top Grant Description Words by USD Cluster":
    st.header("5. Top Grant Description Words by USD Cluster")
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

# 6. Treemaps of Grant Amount by Subject, Population and Strategy
elif selected_chart == "Treemaps of Grant Amount by Subject, Population and Strategy":
    st.header("6. Treemaps of Grant Amount by Subject, Population and Strategy")
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

    # Add summary table of all grants
    st.subheader("All Grants Summary")
    grant_summary = grouped_df[
        ['grant_key', 'amount_usd', 'year_issued', 'grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran']]
    st.write(grant_summary)

# 7. Univariate Analysis of Numeric Columns
elif selected_chart == "Univariate Analysis of Numeric Columns":
    st.header("7. Univariate Analysis of Numeric Columns")
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

# 8. Top Categories by Unique Grant Count
elif selected_chart == "Top Categories by Unique Grant Count":
    st.header("8. Top Categories by Unique Grant Count")
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