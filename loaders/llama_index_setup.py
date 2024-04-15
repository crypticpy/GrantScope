import os
import matplotlib.pyplot as plt
import openai
import tempfile
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Setup your API Key using the loaded environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


# Setup LlamaIndex with specific model and settings
def setup_llama_index():
    Settings.llm = OpenAI(temperature=0.2, model="gpt-4")


# Function to query data using LlamaIndex
def query_data(df, query_text, pre_prompt):
    setup_llama_index()
    full_prompt = pre_prompt + " " + query_text  # Combine pre-prompt with the user's query
    query_engine = PandasQueryEngine(df=df, verbose=True, synthesize_response=True)
    response = query_engine.query(full_prompt)
    return response