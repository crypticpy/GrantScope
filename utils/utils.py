from io import BytesIO
from io import StringIO
import base64
import pandas as pd
import streamlit as st

def download_excel(df, filename):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_csv(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context):
    # Generate a list of available columns in the dataframes
    columns = ', '.join(df.columns)
    grouped_columns = ', '.join(grouped_df.columns)

    # Generate data type information
    data_types = df.dtypes.apply(lambda x: x.name).to_dict()
    data_type_info = ", ".join([f"{col}: {dtype}" for col, dtype in data_types.items()])

    # Generate summary statistics
    summary_stats = df.describe().to_dict()
    summary_stats_info = ", ".join([f"{col}: {stats}" for col, stats in summary_stats.items()])

    # Generate observations about the dataset
    num_records = len(df)
    num_funders = df['funder_name'].nunique()
    num_recipients = df['recip_name'].nunique()
    observations = f"The dataset contains {num_records} records, with {num_funders} unique funders and {num_recipients} unique recipients."

    # Generate a description of the selected chart
    chart_description = f"The current chart is a {selected_chart}, which visualizes the grant data based on {additional_context}."

    # Generate a description of the user's role
    role_description = f"The user is a {selected_role} who is exploring the grant data to gain insights and inform their work."

    # Compose the custom prompt
    prompt = f"The Candid API provides comprehensive data on grants and funding. The current dataset contains the following columns: {columns}. "
    prompt += f"The grouped dataset used for aggregations has the following columns: {grouped_columns}. "
    prompt += f"Data types: {data_type_info}. "
    prompt += f"Summary statistics: {summary_stats_info}. "
    prompt += observations + " "
    prompt += chart_description + " " + role_description
    prompt += " The user can ask questions related to the current chart and the overall grant data to gain insights and explore the data further."
    prompt += " Please note that the data is limited to the information provided in the dataset, so queries beyond the available columns may not be answerable."

    return prompt