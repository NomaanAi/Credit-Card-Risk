import pandas as pd
import os

def load_raw_data(filepath):
    """
    Load raw data from excel file.
    
    Args:
        filepath (str): Path to the raw excel file.
        
    Returns:
        pd.DataFrame: Loaded dataframe with cleaned column names.
    """
    # UCI dataset often has the header in the second row (index 1)
    try:
        df = pd.read_excel(filepath, header=1)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    # Rename columns to snake_case
    df.columns = [c.lower().replace(' ', '_').replace('.', '_') for c in df.columns]
    
    # Rename specific target column if usually named awkwardly
    if 'default_payment_next_month' in df.columns:
        df.rename(columns={'default_payment_next_month': 'target'}, inplace=True)
        
    # Drop ID column if exists
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
        
    return df
