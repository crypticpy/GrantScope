import io
import json
import pandas as pd
import streamlit as st
import numpy as np
from typing import List
from dataclasses import dataclass, asdict

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
    # Convert 'year_issued' to integer, handling missing or invalid values
    df['year_issued'] = pd.to_numeric(df['year_issued'], errors='coerce').fillna(0).astype(int)

    # Handle missing or NaN values for 'amount_usd'
    df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
    df = df.dropna(subset=['amount_usd'])  # Remove rows where 'amount_usd' is NaN

    # Optionally, fill NaNs for other numerical columns with the median
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    # Fill NaNs for categorical columns with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    code_columns = [col for col in df.columns if "_code" in col]
    tran_columns = [col for col in df.columns if "_tran" in col]

    for code_col, tran_col in zip(code_columns, tran_columns):
        df[[code_col, tran_col]] = df[[code_col, tran_col]].map(
            lambda x: x.split(';') if isinstance(x, str) else ['Unknown']
        )
        df = df.explode(code_col).explode(tran_col)

    df = df.fillna({'object': 'Unknown', 'number': df.select_dtypes(include=['number']).median()})
    df['grant_description'] = df['grant_description'].fillna('').astype(str)

    # Create amount_usd clusters
    bins = [0, 50000, 100000, 500000, 1000000, np.inf]
    names = ['0-50k', '50k-100k', '100k-500k', '500k-1M', '1M+']
    df['amount_usd_cluster'] = pd.cut(df['amount_usd'], bins, labels=names)

    # Assuming 'year_issued' and another column (e.g., 'grant_key') uniquely identify rows
    df = df.drop_duplicates(subset=['year_issued', 'grant_key'])

    grouped_df = df.groupby('grant_index').first()

    return df, grouped_df