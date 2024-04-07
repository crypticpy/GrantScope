import base64
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass, asdict
from io import BytesIO
from textwrap import shorten
from typing import List
from wordcloud import WordCloud, STOPWORDS

def clear_cache():
    st.cache_data.clear()

def init_session_state():
    if 'cache_initialized' not in st.session_state:
        st.session_state.cache_initialized = False

    if not st.session_state.cache_initialized:
        clear_cache()
        st.session_state.cache_initialized = True

st.set_page_config(page_title="GrantScope", page_icon=":chart_with_upwards_trend:")

uploaded_file = None

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
def load_data(file_path=None, uploaded_file=None):

    if uploaded_file is not None:
        data = json.load(io.BytesIO(uploaded_file.read()))
    elif file_path is not None:
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        raise ValueError("Either file_path or uploaded_file must be provided.")

    grants = Grants(grants=[Grant(**grant) for grant in data['grants']])
    return grants


@st.cache_data
def preprocess_data(grants):

    df = pd.DataFrame([asdict(grant) for grant in grants.grants])

    df['grant_index'] = df['grant_key']

    code_columns = [col for col in df.columns if "_code" in col]
    tran_columns = [col for col in df.columns if "_tran" in col]

    for code_col, tran_col in zip(code_columns, tran_columns):
        df[[code_col, tran_col]] = df[[code_col, tran_col]].map(
            lambda x: x.split(';') if isinstance(x, str) else ['Unknown']
        )
        df = df.explode(code_col).explode(tran_col)

    df = df.fillna({'object': 'Unknown', 'number': df.select_dtypes(include=['number']).median()})

    bins = [0, 50000, 100000, 500000, 1000000, np.inf]
    names = ['0-50k', '50k-100k', '100k-500k', '500k-1M', '1M+']

    df['amount_usd_cluster'] = pd.cut(df['amount_usd'], bins, labels=names)

    grouped_df = df.groupby('grant_index').first()

    return df, grouped_df


def download_excel(df, filename):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    init_session_state()

file_path = '../loaders/fixed_ovp.json'

uploaded_file = st.sidebar.file_uploader("Upload Candid API JSON File", type="json")

if uploaded_file is not None:
    grants = load_data(uploaded_file=uploaded_file)
else:
    grants = load_data(file_path=file_path)

df, grouped_df = preprocess_data(grants)

chart_options = {
    "Grant Analyst/Writer": [
        "Data Summary",
        "Grant Amount Distribution",
        "Grant Amount Scatter Plot",
        "Grant Amount Heatmap",
        "Grant Description Word Clouds",
        "Treemaps with Extended Analysis",
        "General Analysis of Relationships",
        "Top Categories by Unique Grant Count"
    ],
    "Normal Grant User": [
        "Data Summary",
        "Grant Amount Distribution",
        "Grant Amount Scatter Plot",
        "Grant Amount Heatmap",
        "Grant Description Word Clouds",
        "Treemaps by Subject, Population and Strategy"
    ]
}

user_roles = ["Grant Analyst/Writer", "Normal Grant User"]
selected_role = st.sidebar.selectbox("Select User Role", options=user_roles)
selected_chart = st.sidebar.selectbox("Select Chart", options=chart_options[selected_role])

st.title("GrantScope Dashboard")

if selected_chart == "Data Summary":
    st.header("Data Summary")
    st.write("""
    Welcome to the GrantScope Tool! This powerful application is designed to assist grant writers and analysts in navigating and extracting insights from a comprehensive grant dataset. By leveraging the capabilities of this tool, you can identify potential funding opportunities, analyze trends, and gain valuable information to enhance your grant proposals.

    The preloaded dataset encompasses a wide range of information, including details about funders, recipients, grant amounts, subject areas, populations served, and more. With this tool, you can explore the data through interactive visualizations, filter and search for specific grants, and download relevant data for further analysis.

    """)

    if selected_role == "Grant Analyst/Writer" and selected_chart == "Data Summary":
        st.subheader("Dataset Overview")
        st.write(df.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Unique Grants", value=df['grant_key'].nunique())
    with col2:
        st.metric(label="Total Unique Funders", value=df['funder_name'].nunique())
    with col3:
        st.metric(label="Total Unique Recipients", value=df['recip_name'].nunique())

    st.subheader("Top Funders by Total Grant Amount")
    top_n = st.slider("Select the number of top funders to display", min_value=5, max_value=20, value=10, step=1)
    unique_df = df.drop_duplicates(subset='grant_key')
    top_funders = unique_df.groupby('funder_name')['amount_usd'].sum().nlargest(top_n).reset_index()

    fig = px.bar(top_funders, x='funder_name', y='amount_usd', title=f"Top {top_n} Funders by Total Grant Amount")
    fig.update_layout(xaxis_title='Funder Name', yaxis_title='Total Grant Amount (USD)')
    st.plotly_chart(fig)

    if st.checkbox("Show Top Funders Data Table"):
        st.write(top_funders)

    st.subheader("Grant Distribution by Funder Type")
    funder_type_dist = unique_df.groupby('funder_type')['amount_usd'].sum().reset_index()

    fig = px.pie(funder_type_dist, values='amount_usd', names='funder_type', title="Grant Distribution by Funder Type")
    st.plotly_chart(fig)

    if st.checkbox("Show Funder Type Data Table"):
        st.write(funder_type_dist)

    st.subheader("Grant Distribution by Subject Area")
    subject_dist = unique_df.groupby('grant_subject_tran')['amount_usd'].sum().nlargest(10).reset_index()

    fig = px.bar(subject_dist, x='grant_subject_tran', y='amount_usd',
                 title="Top 10 Grant Subject Areas by Total Amount")
    fig.update_layout(xaxis_title='Subject Area', yaxis_title='Total Grant Amount (USD)')
    st.plotly_chart(fig)

    st.subheader("Grant Distribution by Population Served")
    population_dist = unique_df.groupby('grant_population_tran')['amount_usd'].sum().nlargest(10).reset_index()

    fig = px.bar(population_dist, x='grant_population_tran', y='amount_usd',
                 title="Top 10 Populations Served by Total Grant Amount")
    fig.update_layout(xaxis_title='Population Served', yaxis_title='Total Grant Amount (USD)')
    st.plotly_chart(fig)

    st.write("""
    This page serves as a glimpse into insights you can uncover using GrantScope. Feel free to explore the other plots of the application by using the menu on the left. From there you can dive deeper into specific aspects of the grant data, such as analyzing trends over time or examining population distributions.

    Happy exploring and best of luck with your grant related endeavors!
    
    This app was produced with Candid API, Streamlit, Plotly, and other open-source libraries. Generative AI solution such as Open ai GPT-4 and Claude Opus were used to generate portions of the source code.
    """)
elif selected_chart == "Grant Amount Distribution":
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


elif selected_chart == "Grant Amount Scatter Plot":
    st.header("Grant Amount Scatter Plot")
    st.write("Explore the distribution of grant amounts over time using an interactive scatter plot. The plot will dynamically update based on the selected USD clusters and year range available in the data.")

    unique_years = sorted(df['year_issued'].unique())
    if len(unique_years) == 1:
        unique_year = int(unique_years[0])
        st.write(f"Data available for year: {unique_year}")
        start_year, end_year = unique_year, unique_year
    else:
        start_year = st.number_input("Start Year", min_value=int(min(unique_years)), max_value=int(max(unique_years)),
                                     value=int(min(unique_years)))
        end_year = st.number_input("End Year", min_value=int(min(unique_years)), max_value=int(max(unique_years)),
                                   value=int(max(unique_years)))

    filtered_df = grouped_df[
        (grouped_df['year_issued'].astype(int) >= start_year) &
        (grouped_df['year_issued'].astype(int) <= end_year)
        ]

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

    fig = px.scatter(
        filtered_df,
        x='year_issued',
        y='amount_usd',
        color='amount_usd_cluster',
        hover_data=['grant_key', 'grant_description'],
        opacity=opacity
    )

    fig.update_traces(marker=dict(size=marker_size))

    fig.update_layout(
        title='Grant Amount by Year' if len(unique_years) > 1 else f'Grant Amount for Year {start_year}',
        xaxis_title='Year Issued',
        yaxis_title='Amount (USD)',
        legend_title_text='USD Cluster',
        legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"),
        clickmode='event+select'
    )

    fig.update_xaxes(tickformat='d')
    fig.update_traces(hovertemplate="<b>Grant Key:</b> %{customdata[0]}<br><b>Description:</b> %{customdata[1]}")

    st.plotly_chart(fig)

    if st.checkbox("Show Underlying Data for Chart"):
        st.write(filtered_df)

    if st.button("Download Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="grants_data_chart.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

elif selected_chart == "Grant Amount Heatmap":
    st.header("Grant Amount Heatmap")
    st.write("Explore the intersection of grant dimensions and identify meaningful funding patterns.")

    dimension_options = ['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran']
    default_dim1, default_dim2 = dimension_options[:2]

    col1, col2 = st.columns(2)
    with col1:
        dimension1 = st.selectbox("Select Dimension 1", options=dimension_options,
                                  index=dimension_options.index(default_dim1))
    with col2:
        dimension2 = st.selectbox("Select Dimension 2", options=[d for d in dimension_options if d != dimension1],
                                  index=0)

    st.caption("Select individual values for each dimension to filter the heatmap.")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f"Select {dimension1.split('_')[1].capitalize()}s", expanded=False):
            selected_values1 = st.multiselect(f"Select {dimension1.split('_')[1].capitalize()}s",
                                              options=grouped_df[dimension1].unique(),
                                              default=grouped_df[dimension1].unique())
    with col2:
        with st.expander(f"Select {dimension2.split('_')[1].capitalize()}s", expanded=False):
            selected_values2 = st.multiselect(f"Select {dimension2.split('_')[1].capitalize()}s",
                                              options=grouped_df[dimension2].unique(),
                                              default=grouped_df[dimension2].unique())

    pivot_table = grouped_df[
        grouped_df[dimension1].isin(selected_values1) &
        grouped_df[dimension2].isin(selected_values2)
        ].groupby([dimension1, dimension2])['amount_usd'].sum().unstack().fillna(0)

    fig = go.Figure(data=go.Heatmap(
        x=pivot_table.columns,
        y=pivot_table.index,
        z=pivot_table.values,
        colorscale='Plasma',
        hovertemplate='<b>%{yaxis.title.text}</b>: %{y}<br><b>%{xaxis.title.text}</b>: %{x}<br><b>Total Grant Amount</b>: %{z:,.0f}',
        colorbar=dict(title='Total Grant Amount')
    ))

    fig.update_layout(
        title=f'Total Grant Amount by {dimension1.split("_")[1].capitalize()} and {dimension2.split("_")[1].capitalize()}',
        xaxis_title=dimension2.split('_')[1].capitalize(),
        yaxis_title=dimension1.split('_')[1].capitalize(),
        width=800,
        height=800
    )

    st.plotly_chart(fig)

    if st.checkbox("Show Underlying Data"):
        st.write(pivot_table)

    if st.button("Download Data as Excel"):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        pivot_table.to_excel(writer, sheet_name='Heatmap Data')
        writer.close()
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="heatmap_data.xlsx">Download Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.subheader("Explore Grant Details")
    selected_value1 = st.selectbox(f"Select {dimension1.split('_')[1].capitalize()}", options=selected_values1)

    filtered_df = grouped_df[grouped_df[dimension1] == selected_value1]
    available_values2 = filtered_df[dimension2].unique().tolist()

    if available_values2:

        selected_value2 = st.selectbox(f"Select {dimension2.split('_')[1].capitalize()}", options=available_values2)

        cell_grants = grouped_df[
            (grouped_df[dimension1] == selected_value1) &
            (grouped_df[dimension2] == selected_value2)
            ]

        if not cell_grants.empty:

            st.write(
                f"Grants for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}")
            grant_details = cell_grants[['grant_key', 'grant_description', 'amount_usd']]
            st.write(grant_details)

            if st.button("Download Grant Details as CSV"):
                csv = grant_details.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="grant_details.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:

            st.write(
                f"No grants found for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}")
    else:

        st.write(
            f"No {dimension2.split('_')[1].capitalize()}s available for the selected {dimension1.split('_')[1].capitalize()}.")

elif selected_chart == "Grant Description Word Clouds":
    st.header("Grant Description Word Clouds")
    st.write("Explore the most common words in grant descriptions based on different criteria.")

    cloud_basis_options = ['USD Cluster', 'Funder', 'Population', 'Strategy']
    selected_basis = st.selectbox("Select the basis for generating word clouds:", options=cloud_basis_options)

    column_mapping = {
        'USD Cluster': 'amount_usd_cluster',
        'Funder': 'funder_name',
        'Population': 'grant_population_tran',
        'Strategy': 'grant_strategy_tran'
    }
    selected_column = column_mapping[selected_basis]

    unique_values = grouped_df[selected_column].unique().tolist()

    selected_values = st.multiselect(f"Select {selected_basis}(s)", options=unique_values, default=unique_values)

    stopwords = set(STOPWORDS)
    additional_stopwords = {'public', 'Public', 'health', 'Health', 'and', 'And', 'to', 'To', 'of', 'Of', 'the', 'The',
                            'a', 'A', 'by', 'By', 'in', 'In', 'for', 'For', 'with', 'With', 'on', 'On', 'is', 'Is',
                            'that', 'That', 'are', 'Are', 'as', 'As', 'be', 'Be', 'this', 'This', 'will', 'Will', 'at',
                            'At', 'from', 'From', 'or', 'Or', 'an', 'An', 'which', 'Which', 'have', 'Have', 'it', 'It',
                            'not', 'Not', 'who', 'Who', 'their', 'Their', 'we', 'We', 'support', 'Support', 'project',
                            'Project'}
    stopwords.update(additional_stopwords)

    charts_per_page = 10

    total_pages = (len(selected_values) + charts_per_page - 1) // charts_per_page

    page_number = st.selectbox("Select Page", options=list(range(1, total_pages + 1)))

    start_index = (page_number - 1) * charts_per_page
    end_index = min(start_index + charts_per_page, len(selected_values))

    for value in selected_values[start_index:end_index]:

        filtered_df = grouped_df[grouped_df[selected_column] == value]

        text = ' '.join(filtered_df['grant_description'])

        wordcloud = WordCloud(stopwords=stopwords, width=400, height=200).generate(text)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Word Cloud for {selected_basis}: {value}')
            st.pyplot(fig)
            plt.close(fig)  # Close the figure to free up memory

        with col2:

            words = [word for word in text.split() if word.lower() not in stopwords]
            word_freq = pd.Series(words).value_counts()
            st.write(f"Top Words for {selected_basis}: {value}")
            st.write(word_freq.head(5))

    if st.checkbox("Show Grant Descriptions for Selected Word"):
        selected_word = st.text_input("Enter a word to search in grant descriptions:")

        if selected_word:

            filtered_df = grouped_df[grouped_df[selected_column].isin(selected_values)]
            grant_descriptions = filtered_df[filtered_df['grant_description'].str.contains(selected_word, case=False)]

            if not grant_descriptions.empty:
                st.write(f"Grant Descriptions containing '{selected_word}':")
                for desc in grant_descriptions['grant_description']:
                    st.write(f"- {desc}")
            else:
                st.write(f"No grant descriptions found containing '{selected_word}'.")

elif selected_chart == "Treemaps with Extended Analysis":

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

    grouped_data = filtered_data.groupby(analyze_column)['amount_usd'].sum().reset_index().sort_values(by='amount_usd',
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


elif selected_chart == "General Analysis of Relationships":
    st.header("General Analysis of Relationships")
    st.write("Explore the relationships between various factors and the award amount.")

    unique_grants_df = df.drop_duplicates(subset=['grant_key'])

    st.subheader("Relationship between Grant Description Length and Award Amount")
    unique_grants_df['description_word_count'] = unique_grants_df['grant_description'].apply(
        lambda x: len(str(x).split()))
    fig = px.scatter(unique_grants_df, x='description_word_count', y='amount_usd', opacity=0.5,
                     title="Grant Description Length vs. Award Amount")
    fig.update_layout(xaxis_title='Number of Words in Grant Description', yaxis_title='Award Amount (USD)',
                      width=800, height=600)
    st.plotly_chart(fig)

    st.subheader("Distribution of Award Amounts by Different Factors")
    factors = ['grant_strategy_tran', 'grant_population_tran', 'grant_geo_area_tran', 'funder_name']
    selected_factor = st.selectbox("Select Factor", options=factors)

    exploded_df = unique_grants_df.assign(
        **{selected_factor: unique_grants_df[selected_factor].str.split(';')}).explode(selected_factor)

    fig = px.box(exploded_df, x=selected_factor, y='amount_usd',
                 title=f"Award Amount Distribution by {selected_factor}")
    fig.update_layout(xaxis_title=selected_factor, yaxis_title='Award Amount (USD)',
                      width=800, height=600, boxmode='group')
    st.plotly_chart(fig)

    st.subheader("Average Award Amount by Different Factors")
    avg_amount_by_factor = exploded_df.groupby(selected_factor)['amount_usd'].mean().reset_index()
    avg_amount_by_factor = avg_amount_by_factor.sort_values('amount_usd', ascending=False)
    fig = px.bar(avg_amount_by_factor, x=selected_factor, y='amount_usd',
                 title=f"Average Award Amount by {selected_factor}")
    fig.update_layout(xaxis_title=selected_factor, yaxis_title='Average Award Amount (USD)',
                      width=800, height=600, xaxis_tickangle=-45, xaxis_tickfont=dict(size=10))
    st.plotly_chart(fig)

    st.subheader("Funder Affinity Analysis")
    funders = unique_grants_df['funder_name'].unique().tolist()
    selected_funder = st.selectbox("Select Funder", options=funders)

    affinity_factors = ['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran']
    selected_affinity_factor = st.selectbox("Select Affinity Factor", options=affinity_factors)

    funder_grants_df = unique_grants_df[unique_grants_df['funder_name'] == selected_funder]

    exploded_funder_df = funder_grants_df.assign(
        **{selected_affinity_factor: funder_grants_df[selected_affinity_factor].str.split(';')}).explode(
        selected_affinity_factor)

    funder_affinity = exploded_funder_df.groupby(selected_affinity_factor)['amount_usd'].sum().reset_index()
    funder_affinity = funder_affinity.sort_values('amount_usd', ascending=False)

    fig = px.bar(funder_affinity, x=selected_affinity_factor, y='amount_usd',
                 title=f"Funder Affinity: {selected_funder} - {selected_affinity_factor}")
    fig.update_layout(xaxis_title=selected_affinity_factor, yaxis_title='Total Award Amount (USD)',
                      width=800, height=600, xaxis_tickangle=-45, xaxis_tickfont=dict(size=10))

    st.plotly_chart(fig)

    if st.checkbox("Show Underlying Data"):
        st.write(unique_grants_df)

    if st.button("Download Data as CSV"):
        csv = unique_grants_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="grant_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

elif selected_chart == "Top Categories by Unique Grant Count":

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

if __name__ == '__main__':
    main()