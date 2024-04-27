import base64
from io import BytesIO

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

    # Generate data type information
    data_types = df.dtypes.apply(lambda x: x.name).to_dict()
    data_type_info = ", ".join([f"{col}: {dtype}" for col, dtype in data_types.items()])

    # Generate observations about the dataset
    num_records = len(df)
    num_funders = df['funder_name'].nunique()
    num_recipients = df['recip_name'].nunique()
    observations = f"The dataset contains {num_records} records, with {num_funders} unique funders and {num_recipients} unique recipients."

    # Generate date range information
    min_date = df['last_updated'].min()
    max_date = df['last_updated'].max()
    date_info = f"The dataset covers grants from {min_date} to {max_date}."

    # Generate geographical coverage information
    unique_states = df['funder_state'].unique().tolist()
    geographical_info = f"The dataset covers grants from {len(unique_states)} states in the USA. The top states by grant count are {', '.join(df['funder_state'].value_counts().nlargest(3).index.tolist())}."

    # Generate aggregated statistics
    total_amount = df['amount_usd'].sum()
    avg_amount = df['amount_usd'].mean()
    median_amount = df['amount_usd'].median()
    aggregated_stats = f"The total grant amount is ${total_amount:,.2f}, with an average grant amount of ${avg_amount:,.2f} and a median grant amount of ${median_amount:,.2f}."

    # Generate a description of the selected chart
    chart_description = f"The current chart is a {selected_chart}, which visualizes the grant data based on {additional_context}."

    # Generate a description of the user's role
    role_description = f"The user is a {selected_role} who is exploring the grant data to gain insights and inform their work."

    # Compose the custom per page prompt
    prompt = f"The Candid API provides comprehensive data on grants and funding in the USA. The current dataset contains the following columns: {columns}. "
    prompt += f"You are an AI assistant helping a {selected_role} explore the grant data in the GRantScope application to gain insights and extract data useful to the gran application and writing process. "
    prompt += f"Data types: {data_type_info}. "
    prompt += observations + " "
    prompt += date_info + " "
    prompt += geographical_info + " "
    prompt += aggregated_stats + " "
    prompt += chart_description + " " + role_description
    prompt += " The user can ask questions related to the current chart and the overall grant data to gain insights and explore the data further."
    prompt += " Please note that the data is limited to the information provided in the dataset, queries beyond the available columns are not answerable."
    prompt += " Respond in Markdown format only"
    prompt += " The users prompt is:"

    return prompt
