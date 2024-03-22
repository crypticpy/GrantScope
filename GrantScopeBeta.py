import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textwrap import shorten
import base64
import math
from io import BytesIO
import json
from dataclasses import dataclass, asdict
from typing import List

uploaded_file = None

# Define the Grant and Grants data classes
@dataclass
class Grant:
    """
    A data class that represents a Grant.

    Attributes:
    funder_key (str): The unique identifier for the funder.
    funder_profile_url (str): The URL of the funder's profile.
    funder_name (str): The name of the funder.
    funder_city (str): The city where the funder is located.
    funder_state (str): The state where the funder is located.
    funder_country (str): The country where the funder is located.
    funder_type (str): The type of the funder.
    funder_zipcode (str): The zipcode of the funder's location.
    funder_country_code (str): The country code of the funder's location.
    funder_ein (str): The Employer Identification Number of the funder.
    funder_gs_profile_update_level (str): The update level of the funder's profile.
    recip_key (str): The unique identifier for the recipient.
    recip_name (str): The name of the recipient.
    recip_city (str): The city where the recipient is located.
    recip_state (str): The state where the recipient is located.
    recip_country (str): The country where the recipient is located.
    recip_zipcode (str): The zipcode of the recipient's location.
    recip_country_code (str): The country code of the recipient's location.
    recip_ein (str): The Employer Identification Number of the recipient.
    recip_organization_code (str): The organization code of the recipient.
    recip_organization_tran (str): The organization transaction of the recipient.
    recip_gs_profile_link (str): The link to the recipient's profile.
    recip_gs_profile_update_level (str): The update level of the recipient's profile.
    grant_key (str): The unique identifier for the grant.
    amount_usd (int): The amount of the grant in USD.
    grant_subject_code (str): The subject code of the grant.
    grant_subject_tran (str): The subject transaction of the grant.
    grant_population_code (str): The population code of the grant.
    grant_population_tran (str): The population transaction of the grant.
    grant_strategy_code (str): The strategy code of the grant.
    grant_strategy_tran (str): The strategy transaction of the grant.
    grant_transaction_code (str): The transaction code of the grant.
    grant_transaction_tran (str): The transaction of the grant.
    grant_geo_area_code (str): The geographical area code of the grant.
    grant_geo_area_tran (str): The geographical area transaction of the grant.
    year_issued (str): The year the grant was issued.
    grant_duration (str): The duration of the grant.
    grant_description (str): The description of the grant.
    last_updated (str): The last updated date of the grant.
    """

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
    # The Grants class has a single attribute 'grants' which is a list of Grant objects
    grants: List[Grant]

@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    """
    Function to load grant data from a file or an uploaded file.

    This function reads a JSON file and converts it into a Grants object.
    The Grants object is a list of Grant objects, where each Grant object represents a grant.

    The function uses the @st.cache_data decorator to cache the data,
    so that the data is loaded only once and reused across multiple runs.

    Parameters:
    file_path (str): The path to the JSON file. Default is None.
    uploaded_file (UploadedFile): The uploaded file object. Default is None.

    Returns:
    Grants: A Grants object representing the loaded data.
    """

    # If an uploaded file is provided, load the data from the uploaded file
    if uploaded_file is not None:
        data = json.load(io.BytesIO(uploaded_file.read()))
    else:
        # If a file path is provided, load the data from the file at the given path
        with open(file_path, 'r') as file:
            data = json.load(file)

    # Convert the loaded data into a Grants object
    grants = Grants(grants=[Grant(**grant) for grant in data['grants']])
    return grants


def preprocess_data(grants):
    # This function preprocesses the grant data.
    # It takes a Grants object as an argument and returns a DataFrame.
    # The function performs several preprocessing steps:
    # - It converts the Grants object to a DataFrame.
    # - It creates a 'grant_index' column based on the 'grant_key' column.
    # - It splits and explodes the code and tran columns.
    # - It fills missing values in text columns with 'Unknown' and in numeric columns with the median value.
    # - It creates an 'amount_usd_cluster' column based on the 'amount_usd' column.
    # - It groups the DataFrame by 'grant_index' and takes the first row of each group.
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

# Define the function for downloading an Excel file
def download_excel(df, filename):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Define the file path for the JSON file containing the grant data
file_path = 'fixed_ovp.json'

# Create a file uploader in the sidebar for the user to upload a Candid API JSON file
uploaded_file = st.sidebar.file_uploader("Upload Candid API JSON File", type="json")

# Check if an uploaded file exists in the global scope
if uploaded_file is not None:
    grants = load_data(uploaded_file=uploaded_file)
else:
    grants = load_data(file_path=file_path)

df, grouped_df = preprocess_data(grants)

# Define chart options for each user role
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

# Define the user roles
user_roles = ["Grant Analyst/Writer", "Normal Grant User"]
# Create a select box in the sidebar for the user to select their role
selected_role = st.sidebar.selectbox("Select User Role", options=user_roles)
# Create a select box in the sidebar for the user to select the chart they want to view
# The options are determined by the selected user role
selected_chart = st.sidebar.selectbox("Select Chart", options=chart_options[selected_role])


# Set the title of the main content area to "GrantScope Dashboard"
st.title("GrantScope Dashboard")

# If the user selects "Data Summary" from the chart options
if selected_chart == "Data Summary":
    st.header("Data Summary")
    st.write("""
    Welcome to the GrantScope Tool! This powerful application is designed to assist grant writers and analysts in navigating and extracting insights from a comprehensive grant dataset. By leveraging the capabilities of this tool, you can identify potential funding opportunities, analyze trends, and gain valuable information to enhance your grant proposals.

    The preloaded dataset encompasses a wide range of information, including details about funders, recipients, grant amounts, subject areas, populations served, and more. With this tool, you can explore the data through interactive visualizations, filter and search for specific grants, and download relevant data for further analysis.
    
    """)

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
    This page serves as a glimpse into insights you can uncover using GrantScope. Feel free to explore the other pages of the application by using the menu on the left. From there you can dive deeper into specific aspects of the grant data, such as analyzing trends over time or examining population distributions.

    Happy exploring and best of luck with your grant related endeavors!
    """)
# Grant Amount Distribution by USD Cluster
elif selected_chart == "Grant Amount Distribution":
    st.header("Grant Amount Distribution")
    st.write("Explore the distribution of grant amounts across different USD clusters and apply optional drill-down filters.")

    cluster_options = grouped_df['amount_usd_cluster'].unique().tolist()
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options)

    filtered_df = grouped_df[grouped_df['amount_usd_cluster'].isin(selected_clusters)]


    # create a page divider
    st.divider()
    st.write("It is recommended to use the multi select options above to narrow your scope before applying these drill down filters. Otherwise the chart may take a long time to load, and there will be too many options to sort.")
    col1, col2 = st.columns(2)
    with col1:
        drill_down_options = st.expander("Optional Drill-Down Filters", expanded=False)
        with drill_down_options:
            subject_options = filtered_df['grant_subject_tran'].unique().tolist()
            selected_subjects = st.multiselect("Select Grant Subjects", options=subject_options, default=subject_options,
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
        fig.update_traces(hovertemplate='<b>USD Cluster:</b> %{x}<br><b>Total Grant Amount:</b> %{y}<br><br><b>Grant Key:</b> %{customdata[0]}<br><b>Grant Description:</b> %{customdata[1]}')
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


# Grant Amount vs Year Scatter Plot
elif selected_chart == "Grant Amount Scatter Plot":
    st.header("Grant Amount Scatter Plot")
    st.write("Explore the distribution of grant amounts over time using an interactive scatter plot. The plot will dynamically update based on the selected USD clusters and year range available in the data.")

    unique_years = sorted(df['year_issued'].unique())
    if len(unique_years) == 1:
        unique_year = int(unique_years[0])
        st.write(f"Data available for year: {unique_year}")
        start_year, end_year = unique_year, unique_year
    else:
        start_year = st.number_input("Start Year", min_value=int(min(unique_years)), max_value=int(max(unique_years)), value=int(min(unique_years)))
        end_year = st.number_input("End Year", min_value=int(min(unique_years)), max_value=int(max(unique_years)), value=int(max(unique_years)))

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

    color_scheme = st.selectbox("Color Scheme", options=["viridis", "plasma", "inferno", "magma", "cividis"])
    marker_size = st.slider("Marker Size", min_value=1, max_value=20, value=5)
    opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

    fig = px.scatter(
        filtered_df,
        x='year_issued',
        y='amount_usd',
        color='amount_usd_cluster',
        hover_data=['grant_key', 'grant_description'],
        color_continuous_scale=color_scheme,
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

# Grant Amount by Population and Strategy Heatmap
# This section of the code is executed if the user selects "Grant Amount by Population and Strategy Heatmap".
elif selected_chart == "Grant Amount Heatmap":
    st.header("Grant Amount Heatmap")
    st.write("Explore the intersection of grant dimensions and identify meaningful funding patterns.")

    dimension_options = ['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran']
    default_dim1, default_dim2 = dimension_options[:2]

    col1, col2 = st.columns(2)
    with col1:
        dimension1 = st.selectbox("Select Dimension 1", options=dimension_options, index=dimension_options.index(default_dim1))
    with col2:
        dimension2 = st.selectbox("Select Dimension 2", options=[d for d in dimension_options if d != dimension1], index=0)

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
            st.write(f"Grants for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}")
            grant_details = cell_grants[['grant_key', 'grant_description', 'amount_usd']]
            st.write(grant_details)

            if st.button("Download Grant Details as CSV"):
                csv = grant_details.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="grant_details.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.write(f"No grants found for {dimension1.split('_')[1].capitalize()}: {selected_value1} and {dimension2.split('_')[1].capitalize()}: {selected_value2}")
    else:
        st.write(f"No {dimension2.split('_')[1].capitalize()}s available for the selected {dimension1.split('_')[1].capitalize()}.")

# Top Grant Description Words by USD Cluster
elif selected_chart == "Grant Description Word Clouds":
    st.header("Grant Description Word Clouds")
    st.write("Explore the most common words in grant descriptions based on different criteria.")

    # Define the options for generating word clouds
    cloud_basis_options = ['USD Cluster', 'Funder', 'Population', 'Strategy']
    selected_basis = st.selectbox("Select the basis for generating word clouds:", options=cloud_basis_options)

    # Define the column name mapping based on the selected basis
    column_mapping = {
        'USD Cluster': 'amount_usd_cluster',
        'Funder': 'funder_name',
        'Population': 'grant_population_tran',
        'Strategy': 'grant_strategy_tran'
    }
    selected_column = column_mapping[selected_basis]

    # Get the unique values for the selected column
    unique_values = grouped_df[selected_column].unique().tolist()

    # Create a multi-select box for the user to select one or more values
    selected_values = st.multiselect(f"Select {selected_basis}(s)", options=unique_values, default=unique_values)

    # Define the stopwords and additional stopwords
    stopwords = set(STOPWORDS)
    additional_stopwords = {'public', 'Public', 'health', 'Health', 'and', 'And', 'to', 'To', 'of', 'Of', 'the', 'The',
                            'a', 'A', 'by', 'By', 'in', 'In', 'for', 'For', 'with', 'With', 'on', 'On', 'is', 'Is',
                            'that', 'That', 'are', 'Are', 'as', 'As', 'be', 'Be', 'this', 'This', 'will', 'Will', 'at',
                            'At', 'from', 'From', 'or', 'Or', 'an', 'An', 'which', 'Which', 'have', 'Have', 'it', 'It',
                            'not', 'Not', 'who', 'Who', 'their', 'Their', 'we', 'We', 'support', 'Support', 'project',
                            'Project'}
    stopwords.update(additional_stopwords)

    # Set the number of charts to display per page
    charts_per_page = 10

    # Calculate the total number of pages
    total_pages = (len(selected_values) + charts_per_page - 1) // charts_per_page

    # Create a select box for the user to choose the page number
    page_number = st.selectbox("Select Page", options=list(range(1, total_pages + 1)))

    # Calculate the start and end index of the selected values for the current page
    start_index = (page_number - 1) * charts_per_page
    end_index = min(start_index + charts_per_page, len(selected_values))

    # Display the word clouds and top words for the selected values on the current page
    for value in selected_values[start_index:end_index]:
        # Filter the dataframe based on the selected value
        filtered_df = grouped_df[grouped_df[selected_column] == value]

        # Join the grant descriptions into a single string
        text = ' '.join(filtered_df['grant_description'])

        # Generate the word cloud
        wordcloud = WordCloud(stopwords=stopwords, width=400, height=200).generate(text)

        # Display the word cloud and top words in a single column
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Word Cloud for {selected_basis}: {value}')
            st.pyplot(fig)
            plt.close(fig)  # Close the figure to free up memory

        with col2:
            # Display the top words for the current value
            words = [word for word in text.split() if word.lower() not in stopwords]
            word_freq = pd.Series(words).value_counts()
            st.write(f"Top Words for {selected_basis}: {value}")
            st.write(word_freq.head(5))

    # Provide an option to show grant descriptions for a selected word
    if st.checkbox("Show Grant Descriptions for Selected Word"):
        selected_word = st.text_input("Enter a word to search in grant descriptions:")

        if selected_word:
            # Filter the dataframe based on the selected values and search for the selected word
            filtered_df = grouped_df[grouped_df[selected_column].isin(selected_values)]
            grant_descriptions = filtered_df[filtered_df['grant_description'].str.contains(selected_word, case=False)]

            if not grant_descriptions.empty:
                st.write(f"Grant Descriptions containing '{selected_word}':")
                for desc in grant_descriptions['grant_description']:
                    st.write(f"- {desc}")
            else:
                st.write(f"No grant descriptions found containing '{selected_word}'.")

# Treemaps of Grant Amount by Subject, Population and Strategy
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

    # Initialize 'block_grants' as an empty DataFrame to ensure it's always defined
    block_grants = pd.DataFrame()

    # Only attempt to show detail view if there's grouped data available
    if not grouped_data.empty:
        selected_block = st.selectbox("Select Detail to View",
                                      options=[None] + sorted(grouped_data[analyze_column].unique()), index=0)
        if selected_block:
            block_grants = filtered_data[filtered_data[analyze_column] == selected_block]

            # Display Dashboard Summary in Columns if block grants are not empty
            if not block_grants.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Total Grants", value=len(block_grants))
                with col2:
                    st.metric(label="Total USD", value=f"${block_grants['amount_usd'].sum():,.2f}")
                with col3:
                    st.metric(label="Average Grant USD", value=f"${block_grants['amount_usd'].mean():,.2f}")

                # Sort and Display Detailed View
                block_grants_sorted = block_grants.sort_values(by='amount_usd', ascending=False)
                block_grants_sorted = block_grants_sorted[['grant_description', 'amount_usd']]
                st.subheader("Selected Grant Descriptions")
                st.dataframe(block_grants_sorted)

        # Ensure block_grants is not empty before trying to display the grant summary
        if not block_grants.empty:
            st.subheader("Raw Data for Selected Detail")
            grant_summary = block_grants[
                ['grant_key', 'amount_usd', 'year_issued', 'grant_subject_tran', 'grant_population_tran',
                 'grant_strategy_tran']]
            st.dataframe(grant_summary)
        else:
            st.write("No detailed data available for the selected options.")

    # Download button
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

    # Create a new DataFrame with unique grant keys
    unique_grants_df = df.drop_duplicates(subset=['grant_key'])

    # Analyze the relationship between the number of words in grant descriptions and award amounts
    st.subheader("Relationship between Grant Description Length and Award Amount")
    unique_grants_df['description_word_count'] = unique_grants_df['grant_description'].apply(
        lambda x: len(str(x).split()))
    fig = px.scatter(unique_grants_df, x='description_word_count', y='amount_usd', opacity=0.5,
                     title="Grant Description Length vs. Award Amount")
    fig.update_layout(xaxis_title='Number of Words in Grant Description', yaxis_title='Award Amount (USD)',
                      width=800, height=600)
    st.plotly_chart(fig)

    # Analyze the distribution of award amounts by different factors
    st.subheader("Distribution of Award Amounts by Different Factors")
    factors = ['grant_strategy_tran', 'grant_population_tran', 'grant_geo_area_tran', 'funder_name']
    selected_factor = st.selectbox("Select Factor", options=factors)

    # Explode the selected factor column and create a new DataFrame
    exploded_df = unique_grants_df.assign(
        **{selected_factor: unique_grants_df[selected_factor].str.split(';')}).explode(selected_factor)

    fig = px.box(exploded_df, x=selected_factor, y='amount_usd',
                 title=f"Award Amount Distribution by {selected_factor}")
    fig.update_layout(xaxis_title=selected_factor, yaxis_title='Award Amount (USD)',
                      width=800, height=600, boxmode='group')
    st.plotly_chart(fig)

    # Analyze the average award amount by different factors
    st.subheader("Average Award Amount by Different Factors")
    avg_amount_by_factor = exploded_df.groupby(selected_factor)['amount_usd'].mean().reset_index()
    avg_amount_by_factor = avg_amount_by_factor.sort_values('amount_usd', ascending=False)
    fig = px.bar(avg_amount_by_factor, x=selected_factor, y='amount_usd',
                 title=f"Average Award Amount by {selected_factor}")
    fig.update_layout(xaxis_title=selected_factor, yaxis_title='Average Award Amount (USD)',
                      width=800, height=600, xaxis_tickangle=-45, xaxis_tickfont=dict(size=10))
    st.plotly_chart(fig)

    # Analyze the affinity of funders towards specific grant types, populations, or strategies
    st.subheader("Funder Affinity Analysis")
    funders = unique_grants_df['funder_name'].unique().tolist()
    selected_funder = st.selectbox("Select Funder", options=funders)

    affinity_factors = ['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran']
    selected_affinity_factor = st.selectbox("Select Affinity Factor", options=affinity_factors)

    # Filter the unique grants DataFrame by the selected funder
    funder_grants_df = unique_grants_df[unique_grants_df['funder_name'] == selected_funder]

    # Explode the selected affinity factor column
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

    # Display the underlying data for further analysis
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
            grant_details = category_grants[['grant_key', 'grant_description', 'amount_usd', 'recip_organization_tran', 'year_issued']]
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