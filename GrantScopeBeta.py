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
        "Treemaps by Subject, Population and Strategy",
        "Univariate Analysis of Numeric Columns",
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


# Set the title of the main content area to "Grant Analysis Dashboard"
st.title("Grant Analysis Dashboard")

# If the user selects "Data Summary" from the chart options
if selected_chart == "Data Summary":
    # Set the header of the Streamlit page to "1. Data Summary"
    st.header("Data Summary")

    # Display the first few rows of the DataFrame
    st.write(df.head())

    # Calculate and display the number of unique grants
    unique_grant_keys = df['grant_key'].nunique()
    st.write(f"Total Unique Grants: {unique_grant_keys}")

    # Calculate and display the number of unique funders
    st.write(f"Total Unique Funders: {df['funder_name'].nunique()}")

    # Calculate and display the number of unique recipients
    st.write(f"Total Unique Recipients: {df['recip_name'].nunique()}")

    # Set the header of the Streamlit page to "Top Funders by Total Grant Amount"
    st.header("Top Funders by Total Grant Amount")

    # Create a slider in Streamlit for the user to select the number of top funders to display
    top_n = st.slider("Select the number of top funders to display", min_value=5, max_value=20, value=10, step=1)

    # Drop duplicates based on 'grant_key' to consider each grant only once
    unique_df = df.drop_duplicates(subset='grant_key')

    # Group the unique DataFrame by 'funder_name', calculate the total 'amount_usd' for each group,
    # sort the resulting Series in descending order, and take the top 'top_n' rows
    top_funders = unique_df.groupby('funder_name')['amount_usd'].sum().nlargest(top_n).reset_index()

    # Create a bar chart of the top funders by total grant amount using Plotly Express
    fig = px.bar(top_funders, x='funder_name', y='amount_usd', title=f"Top {top_n} Funders by Total Grant Amount")

    # Update the layout of the figure
    fig.update_layout(xaxis_title='Funder Name', yaxis_title='Total Grant Amount (USD)')

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    # If the user checks the checkbox to show the data table, display the 'top_funders' DataFrame in Streamlit
    if st.checkbox("Show Top Funders Data Table"):
        st.write(top_funders)

    # Set the header of the Streamlit page to "Grant Distribution by Funder Type"
    st.header("Grant Distribution by Funder Type")

    # Drop duplicates based on 'grant_key' to consider each grant only once
    unique_df = df.drop_duplicates(subset='grant_key')

    # Group the unique DataFrame by 'funder_type' and calculate the total 'amount_usd' for each group
    funder_type_dist = unique_df.groupby('funder_type')['amount_usd'].sum().reset_index()

    # Create a pie chart of the grant distribution by funder type using Plotly Express
    fig = px.pie(funder_type_dist, values='amount_usd', names='funder_type', title="Grant Distribution by Funder Type")

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    if st.checkbox("Show Funder Type Data Table"):
        st.write(funder_type_dist)
# Grant Amount Distribution by USD Cluster
# This section of the code is executed if the user selects "Grant Amount Distribution by USD Cluster" from the chart options.
elif selected_chart == "Grant Amount Distribution":
    # Set the header of the Streamlit page to "Grant Amount Distribution by USD Cluster"
    st.header("Grant Amount Distribution")

    # Get a list of unique values in the 'amount_usd_cluster' column of the grouped_df DataFrame
    cluster_options = grouped_df['amount_usd_cluster'].unique().tolist()

    # Create a multi-select box in Streamlit for the user to select one or more USD clusters
    # The default selection is all clusters
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options)

    # Filter the DataFrame to include only rows where 'amount_usd_cluster' is in the selected clusters
    filtered_df = grouped_df[grouped_df['amount_usd_cluster'].isin(selected_clusters)]

    # Create an expander in Streamlit for optional drill-down filters
    drill_down_options = st.expander("Optional Drill-Down Filters", expanded=False)
    with drill_down_options:
        # Get a list of unique values in the 'grant_subject_tran' column of the filtered_df DataFrame
        subject_options = filtered_df['grant_subject_tran'].unique().tolist()
        # Create a multi-select box in Streamlit for the user to select one or more grant subjects
        # The default selection is all subjects
        selected_subjects = st.multiselect("Select Grant Subjects", options=subject_options, default=subject_options,
                                           key='subject_select')
        # Create a button in Streamlit for the user to apply the selected grant subject filters
        subject_submit = st.button("Apply Grant Subject Filters")

        # Get a list of unique values in the 'grant_population_tran' column of the filtered_df DataFrame
        population_options = filtered_df['grant_population_tran'].unique().tolist()
        # Create a multi-select box in Streamlit for the user to select one or more grant populations
        # The default selection is all populations
        selected_populations = st.multiselect("Select Grant Populations", options=population_options,
                                              default=population_options, key='population_select')
        # Create a button in Streamlit for the user to apply the selected grant population filters
        population_submit = st.button("Apply Grant Population Filters")

        # Get a list of unique values in the 'grant_strategy_tran' column of the filtered_df DataFrame
        strategy_options = filtered_df['grant_strategy_tran'].unique().tolist()
        # Create a multi-select box in Streamlit for the user to select one or more grant strategies
        # The default selection is all strategies
        selected_strategies = st.multiselect("Select Grant Strategies", options=strategy_options,
                                             default=strategy_options, key='strategy_select')
        # Create a button in Streamlit for the user to apply the selected grant strategy filters
        strategy_submit = st.button("Apply Grant Strategy Filters")

    # If the user has clicked the button to apply the grant subject filters, use the selected subjects
    # Otherwise, use all subjects
    filtered_subjects = selected_subjects if subject_submit else subject_options
    # If the user has clicked the button to apply the grant population filters, use the selected populations
    # Otherwise, use all populations
    filtered_populations = selected_populations if population_submit else population_options
    # If the user has clicked the button to apply the grant strategy filters, use the selected strategies
    # Otherwise, use all strategies
    filtered_strategies = selected_strategies if strategy_submit else strategy_options

    # Filter the DataFrame to include only rows where 'grant_subject_tran' is in the filtered subjects,
    # 'grant_population_tran' is in the filtered populations, and 'grant_strategy_tran' is in the filtered strategies
    filtered_df = filtered_df[
        (filtered_df['grant_subject_tran'].isin(filtered_subjects)) &
        (filtered_df['grant_population_tran'].isin(filtered_populations)) &
        (filtered_df['grant_strategy_tran'].isin(filtered_strategies))
        ]

    # Create a bar chart using Plotly Express with 'amount_usd_cluster' as the x-axis, 'amount_usd' as the y-axis,
    # and 'amount_usd_cluster' as the color dimension
    fig = px.bar(filtered_df, x='amount_usd_cluster', y='amount_usd', color='amount_usd_cluster')

    # Add click event to the chart to show underlying data
    fig.update_layout(clickmode='event+select')
    # Update the traces of the chart to have a blue line color, a line width of 1.5, and an opacity of 0.6
    fig.update_traces(selector=dict(type='bar'), marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Check if the user has checked the "Show Grant Descriptions for Selected Filters" checkbox
    if st.checkbox("Show Grant Descriptions for Selected Filters"):
        # If the checkbox is checked, write "Grant Descriptions:" to the Streamlit page
        st.write("Grant Descriptions:")
        # Iterate over each row in the filtered DataFrame
        for _, row in filtered_df.iterrows():
            # For each row, write the grant key and grant description to the Streamlit page
            st.write(f"- Grant Key: {row['grant_key']}")
            st.write(f"  Description: {row['grant_description']}")
            # Write a separator line to the Streamlit page
            st.write("---")

    # If the user checks the checkbox to show the underlying data for the chart, display the filtered DataFrame in Streamlit
    if st.checkbox("Show Underlying Data for Chart"):
        st.write(filtered_df)

    # If the user clicks the button to download data for the chart
    if st.button("Download Data for Chart"):
        # Create a BytesIO object to hold the Excel data
        output = BytesIO()
        # Create a Pandas ExcelWriter object with the BytesIO object as the target
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        # Write the filtered DataFrame to the ExcelWriter object
        filtered_df.to_excel(writer, index=False, sheet_name='Sheet1')
        # Close the ExcelWriter object
        writer.close()
        # Reset the position of the BytesIO object to the beginning
        output.seek(0)
        # Encode the BytesIO object as base64
        b64 = base64.b64encode(output.read()).decode()
        # Create a download link for the Excel file
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_chart.xlsx">Download Excel File</a>'
        # Display the download link in Streamlit
        st.markdown(href, unsafe_allow_html=True)


# Grant Amount vs Year Scatter Plot
elif selected_chart == "Grant Amount Scatter Plot":
    st.header("Grant Amount Scatter Plot")

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
    # Update the hover template to display grant key and description
    fig.update_traces(hovertemplate="<b>Grant Key:</b> %{customdata[0]}<br><b>Description:</b> %{customdata[1]}")
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
# This section of the code is executed if the user selects "Grant Amount by Population and Strategy Heatmap" from the chart options.
elif selected_chart == "Grant Amount Heatmap":
    # Set the header of the Streamlit page to "Grant Amount by Population and Strategy Heatmap"
    st.header("Grant Amount Heatmap")

    # Create a select box in Streamlit for the user to select the first dimension for the heatmap
    # The options are 'grant_subject_tran', 'grant_population_tran', and 'grant_strategy_tran'
    dimension1 = st.selectbox("Select Dimension 1",
                              options=['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran'])
    # Create a select box in Streamlit for the user to select the second dimension for the heatmap
    # The options are 'grant_subject_tran', 'grant_population_tran', and 'grant_strategy_tran'
    dimension2 = st.selectbox("Select Dimension 2",
                              options=['grant_subject_tran', 'grant_population_tran', 'grant_strategy_tran'], index=1)

    # Create a multi-select box in Streamlit for the user to select the values for the first dimension
    selected_values1 = st.multiselect(f"Select {dimension1.split('_')[1].capitalize()}s",
                                      options=grouped_df[dimension1].unique(), default=grouped_df[dimension1].unique())
    # Create a multi-select box in Streamlit for the user to select the values for the second dimension
    selected_values2 = st.multiselect(f"Select {dimension2.split('_')[1].capitalize()}s",
                                      options=grouped_df[dimension2].unique(), default=grouped_df[dimension2].unique())

    # Create a pivot table for the heatmap by grouping the DataFrame by the selected dimensions and summing the 'amount_usd' for each group
    # Fill any missing values with 0
    pivot_table = grouped_df[
        grouped_df[dimension1].isin(selected_values1) &
        grouped_df[dimension2].isin(selected_values2)
        ].groupby([dimension1, dimension2])['amount_usd'].sum().unstack().fillna(0)

    # Create a Plotly heatmap with the pivot table data
    fig = go.Figure(data=go.Heatmap(
        x=pivot_table.columns,
        y=pivot_table.index,
        z=pivot_table.values,
        colorscale='Plasma'
    ))

    # Update the layout of the heatmap
    fig.update_layout(
        title=f'Total Grant Amount by {dimension1.split("_")[1].capitalize()} and {dimension2.split("_")[1].capitalize()}',
        xaxis_title=dimension2.split('_')[1].capitalize(),
        yaxis_title=dimension1.split('_')[1].capitalize(),
        width=800,
        height=800
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)

    # If the user clicks the button to download data for the chart, create an Excel file with the data and provide a download link
    if st.button("Download Data for Chart"):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        grouped_df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_chart.xlsx">Download Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Check if the user has checked the "Dive Grant Cross Section Data" checkbox
    if st.checkbox("Dive Grant Cross Section Data"):
        # If the checkbox is checked, create a select box for the user to select a value for the first dimension
        selected_value1 = st.selectbox(f"Select {dimension1.split('_')[1].capitalize()}", options=selected_values1)

        # Filter the DataFrame to include only rows where the first dimension is equal to the selected value
        filtered_df = grouped_df[grouped_df[dimension1] == selected_value1]
        # Get a list of unique values in the second dimension of the filtered DataFrame
        available_values2 = filtered_df[dimension2].unique().tolist()

        # If there are available values for the second dimension
        if available_values2:
            # Create a select box for the user to select a value for the second dimension
            selected_value2 = st.selectbox(f"Select {dimension2.split('_')[1].capitalize()}", options=available_values2)
            # Filter the DataFrame to include only rows where the first dimension is equal to
            # the selected value for the first dimension
            # and the second dimension is equal to the selected value for the second dimension
            cell_grants = grouped_df[
                (grouped_df[dimension1] == selected_value1) &
                (grouped_df[dimension2] == selected_value2)
                ]
            # Write "Grant Descriptions:" to the Streamlit page
            st.write("Grant Descriptions:")
            # Iterate over each row in the filtered DataFrame
            for _, row in cell_grants.iterrows():
                # For each row, write the grant key and grant description to the Streamlit page
                st.write(f"- Grant Key: {row['grant_key']}")
                st.write(f"  Description: {row['grant_description']}")
                # Write a separator line to the Streamlit page
                st.write("---")
        else:
            # If there are no available values for the second dimension, write a message to the Streamlit page
            st.write(f"No {dimension2.split('_')[1]}s available for the selected {dimension1.split('_')[1]}.")

# Top Grant Description Words by USD Cluster
# This section of the code is executed if the user selects
# "Top Grant Description Words by USD Cluster" from the chart options.
elif selected_chart == "Grant Description Word Clouds":
    # Set the header of the Streamlit page to "Top Grant Description Words by USD Cluster"
    st.header("Grant Description Word Clouds")

    # Get a list of unique values in the 'amount_usd_cluster' column of the grouped_df DataFrame
    cluster_options = grouped_df['amount_usd_cluster'].unique().tolist()

    # Create a multi-select box in Streamlit for the user to select one or more USD clusters
    # The default selection is all clusters
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options,
                                       key='wordcloud_clusters')

    # Define a set of common words to be excluded from the word cloud
    stopwords = set(STOPWORDS)

    # Define a set of additional words to be excluded from the word cloud
    additional_stopwords = {'public', 'Public', 'health', 'Health', 'and', 'And', 'to', 'To', 'of', 'Of', 'the', 'The',
                            'a', 'A', 'by', 'By', 'in', 'In', 'for', 'For', 'with', 'With', 'on', 'On', 'is', 'Is',
                            'that', 'That', 'are', 'Are', 'as', 'As', 'be', 'Be', 'this', 'This', 'will', 'Will', 'at',
                            'At', 'from', 'From', 'or', 'Or', 'an', 'An', 'which', 'Which', 'have', 'Have', 'it', 'It',
                            'not', 'Not', 'who', 'Who', 'their', 'Their', 'we', 'We', 'support', 'Support', 'project',
                            'Project'}

    # Add the additional stopwords to the set of stopwords
    stopwords.update(additional_stopwords)

    # Iterate over each cluster in the selected clusters
    for cluster in selected_clusters:
        # Filter the dataframe to get the top 20 grants based on the amount_usd for the current cluster
        top_grants = grouped_df[grouped_df['amount_usd_cluster'] == cluster].nlargest(20, 'amount_usd')
        # Join the grant descriptions of the top grants into a single string
        text = ' '.join(top_grants['grant_description'])

        # Generate a word cloud from the text
        wordcloud = WordCloud(stopwords=stopwords, width=800, height=400).generate(text)

        # Create a new figure and axis for the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        # Display the word cloud on the axis
        ax.imshow(wordcloud, interpolation='bilinear')
        # Hide the axis
        ax.axis('off')
        # Set the title of the plot
        ax.set_title(f'Word Cloud of Grant Descriptions for {cluster} Cluster')
        # Display the plot in Streamlit
        st.pyplot(fig)

        # Create a list of words from the text, excluding stopwords
        words = [word for word in text.split() if word.lower() not in stopwords]
        # Count the frequency of each word
        word_freq = pd.Series(words).value_counts()
        # Display the top words for the current cluster
        st.write(f"Top Words for {cluster} Cluster:")
        # Display the frequency of the top 20 words
        st.write(word_freq.head(20))

        # Check if the user has checked the "Show Grant Descriptions for Selected Word in {cluster} Cluster" checkbox
        if st.checkbox(f"Show Grant Descriptions for Selected Word in {cluster} Cluster"):
            # If the checkbox is checked, create a select box for the user to select a word from the current cluster
            selected_word = st.selectbox(f"Select a word from the {cluster} Cluster", options=list(word_freq.index))

            # Filter the top grants DataFrame to include only rows where the grant description contains the selected word
            # The case of the word is ignored in the search
            grant_descriptions = top_grants[top_grants['grant_description'].str.contains(selected_word, case=False)]

            # Write a header to the Streamlit page with the selected word and current cluster
            st.write(f"Grant Descriptions containing '{selected_word}' in {cluster} Cluster:")
            # Iterate over each grant description in the filtered DataFrame
            for desc in grant_descriptions['grant_description']:
                # For each grant description, write the description to the Streamlit page
                st.write(f"- {desc}")

# Treemaps of Grant Amount by Subject, Population and Strategy
# This section of the code is executed if the user selects "Treemaps of Grant Amount by Subject, Population and Strategy" from the chart options.
elif selected_chart == "Treemaps by Subject, Population and Strategy":
    # Set the header of the Streamlit page to "Treemaps of Grant Amount by Subject, Population and Strategy"
    st.header("Treemaps by Subject, Population and Strategy")

    # Create a radio button selection in Streamlit for the user to select a variable for the treemap.
    # The options are 'grant_strategy_tran', 'grant_subject_tran', and 'grant_population_tran'.
    analyze_column = st.radio("Select Variable for Treemap",
                              options=['grant_strategy_tran', 'grant_subject_tran', 'grant_population_tran'])

    # Iterate over each unique value in the 'amount_usd_cluster' column of the grouped_df DataFrame
    for label in grouped_df['amount_usd_cluster'].unique():
        # Filter the DataFrame to include only rows where 'amount_usd_cluster' equals the current label
        filtered_data = grouped_df[grouped_df['amount_usd_cluster'] == label]
        # Group the filtered DataFrame by the 'analyze_column' and calculate the sum of 'amount_usd' for each group
        # Reset the index of the resulting DataFrame and sort it by 'amount_usd' in descending order
        grouped_data = filtered_data.groupby(analyze_column)['amount_usd'].sum().reset_index().sort_values(
            by='amount_usd', ascending=False)

        # Create a treemap using Plotly Express with the path set to 'analyze_column' and values set to 'amount_usd'
        # The title of the treemap is dynamically generated based on 'analyze_column' and the current label
        fig = px.treemap(grouped_data, path=[analyze_column], values='amount_usd',
                         title=f"Treemap: Sum of Amount in USD by {analyze_column} for {label} USD range")
        # Display the treemap in Streamlit
        st.plotly_chart(fig)

        # Check if the user has checked the checkbox to show grants for the selected block in 'analyze_column'
        # The checkbox label is dynamically generated based on 'analyze_column'
        # The key for the checkbox is a combination of the current label and 'analyze_column'
        if st.checkbox(f"Show Grant Breakdown for {analyze_column} Blocks", key=f"{label}_{analyze_column}"):
            # If the checkbox is checked, create a select box for the user to select a block
            # The options for the select box are the unique values in the 'analyze_column' of the grouped_data DataFrame
            selected_block = st.selectbox(f"Select {analyze_column} Block", options=grouped_data[analyze_column])
            # Filter the filtered_data DataFrame to include only rows where 'analyze_column' equals the selected block
            block_grants = filtered_data[filtered_data[analyze_column] == selected_block]
            # Write "Grant Descriptions:" to the Streamlit page
            st.write("Grant Descriptions:")
            # Iterate over each row in the block_grants DataFrame
            for _, row in block_grants.iterrows():
                # For each row, write the grant key and grant description to the Streamlit page
                st.write(f"- Grant Key: {row['grant_key']}")
                st.write(f"  Description: {row['grant_description']}")
                st.write(f"  Amount (USD): {row['amount_usd']}")
                # Write a separator line to the Streamlit page
                st.write("---")

        # If the user clicks the button to download data for the current USD range
        if st.button(f"Download Data for {label} USD range", key=f"download_{label}"):
            # Create a BytesIO object to hold the Excel data
            output = BytesIO()
            # Create a Pandas ExcelWriter object with the BytesIO object as the target
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            # Write the filtered DataFrame to the ExcelWriter object
            filtered_data.to_excel(writer, index=False, sheet_name='Sheet1')
            # Close the ExcelWriter object
            writer.close()
            # Reset the position of the BytesIO object to the beginning
            output.seek(0)
            # Encode the BytesIO object as base64
            b64 = base64.b64encode(output.read()).decode()
            # Create a download link for the Excel file
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_{label}.xlsx">Download Excel File</a>'
            # Display the download link in Streamlit
            st.markdown(href, unsafe_allow_html=True)

    # Add summary table of all grants
    st.subheader("All Grants Summary")
    grant_summary = grouped_df[
        ['grant_key', 'amount_usd', 'year_issued', 'grant_subject_tran', 'grant_population_tran',
         'grant_strategy_tran']]
    st.write(grant_summary)

# Univariate Analysis of Numeric Columns
elif selected_chart == "Univariate Analysis of Numeric Columns":
    # Set the header of the Streamlit page to "Univariate Analysis of Numeric Columns"
    st.header("Univariate Analysis of Numeric Columns")

    # Get a list of all numeric columns in the DataFrame
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # If 'amount_usd_cluster' is in the list of numeric columns, remove it
    if 'amount_usd_cluster' in numeric_columns:
        numeric_columns.remove('amount_usd_cluster')

    # Create a radio button selection in Streamlit for the user to select a numeric variable
    selected_numeric = st.radio("Select Numeric Variable", options=numeric_columns)

    # Create a histogram of the selected numeric variable using Plotly Express
    # The number of bins is set to 50
    # The title of the histogram is dynamically generated based on the selected numeric variable
    fig = px.histogram(df, x=selected_numeric, nbins=50, title=f"Distribution of {selected_numeric}")
    # Display the histogram in Streamlit
    st.plotly_chart(fig)

    # Create a boxplot of the selected numeric variable using Plotly Express
    # The title of the boxplot is dynamically generated based on the selected numeric variable
    fig = px.box(df, y=selected_numeric, title=f"Boxplot of {selected_numeric}")
    # Display the boxplot in Streamlit
    st.plotly_chart(fig)

    # Display the summary statistics for the selected numeric variable in Streamlit
    st.write(f"Summary Statistics for {selected_numeric}:")
    st.write(df[selected_numeric].describe())

# Top Categories by Unique Grant Count
elif selected_chart == "Top Categories by Unique Grant Count":
    # Set the header of the Streamlit page to "Top Categories by Unique Grant Count"
    st.header("Top Categories by Unique Grant Count")

    # Define the key categorical columns to be used in the analysis
    key_categorical_columns = ['funder_type', 'recip_organization_tran', 'grant_subject_tran', 'grant_population_tran',
                               'grant_strategy_tran', 'year_issued']

    # Create a select box in Streamlit for the user to select a categorical variable
    selected_categorical = st.selectbox("Select Categorical Variable", options=key_categorical_columns)

    # Group the DataFrame by the selected categorical variable and count the unique 'grant_key' values
    # Sort the resulting Series in descending order and reset the index
    normalized_counts = df.groupby(selected_categorical)['grant_key'].nunique().sort_values(
        ascending=False).reset_index()

    # Rename the columns of the DataFrame
    normalized_counts.columns = [selected_categorical, 'Unique Grant Keys']

    # Create a new column 'truncated_col' in the DataFrame that contains the values of the selected categorical variable
    # truncated to a maximum width of 30 characters
    normalized_counts['truncated_col'] = normalized_counts[selected_categorical].apply(
        lambda x: shorten(x, width=30, placeholder="..."))

    # Create a horizontal bar chart of the top 10 categories in the selected categorical variable
    fig = px.bar(normalized_counts.head(10), x='Unique Grant Keys', y='truncated_col', orientation='h',
                 title=f"Top 10 Categories in {selected_categorical}")

    # Update the layout of the figure
    fig.update_layout(yaxis_title=selected_categorical)

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    # Display the percentage of total unique grants that the top 10 categories account for
    st.write(
        f"Top 10 Categories account for {normalized_counts.head(10)['Unique Grant Keys'].sum() / normalized_counts['Unique Grant Keys'].sum():.2%} of total unique grants")

    # Check if the user has checked the "Show Grants for Selected {selected_categorical} Category" checkbox
    # The checkbox label is dynamically generated based on 'selected_categorical'
    if st.checkbox(f"Show Grants for Selected {selected_categorical} Category"):
        # If the checkbox is checked, create a select box for the user to select a category
        # The options for the select box are the truncated categories in the 'normalized_counts' DataFrame
        selected_category = st.selectbox(f"Select {selected_categorical} Category",
                                         options=normalized_counts['truncated_col'])

        # Filter the DataFrame to include only rows where 'selected_categorical' equals the selected category
        category_grants = df[df[selected_categorical] == selected_category]

        # Write "Grant Descriptions:" to the Streamlit page
        st.write("Grant Descriptions:")
        # Iterate over each row in the filtered DataFrame
        for _, row in category_grants.iterrows():
            # For each row, write the grant key and grant description to the Streamlit page
            st.write(f"- Grant Key: {row['grant_key']}")
            st.write(f"  Description: {row['grant_description']}")
            # Write a separator line to the Streamlit page
            st.write("---")

    # If the user clicks the button to download data for the chart
    if st.button("Download Data for Chart"):
        # Create a BytesIO object to hold the Excel data
        output = BytesIO()

        # Create a Pandas ExcelWriter object with the BytesIO object as the target
        writer = pd.ExcelWriter(output, engine='xlsxwriter')

        # Write the DataFrame to the ExcelWriter object
        df.to_excel(writer, index=False, sheet_name='Sheet1')

        # Close the ExcelWriter object
        writer.close()

        # Reset the position of the BytesIO object to the beginning
        output.seek(0)

        # Encode the BytesIO object as base64
        b64 = base64.b64encode(output.read()).decode()

        # Create a download link for the Excel file
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="grants_data_chart.xlsx">Download Excel File</a>'

        # Display the download link in Streamlit
        st.markdown(href, unsafe_allow_html=True)
