import pandas as pd
import requests
from sqlalchemy import create_engine

def load_csv(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def load_api(api_url, params):
    """
    Load data from an API using a GET request.
    """
    response = requests.get(api_url, params=params)

    response.raise_for_status()

    return response.json()

def load_database(database_url, query):
    """
    Load data from a PostgreSQL database.
    """
    engine = create_engine(database_url)

    return pd.read_sql(query, con=engine)

