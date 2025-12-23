import pandas as pd
import os

filepath = "data/raw/credit_default_uci.xls"
# Try reading with default header = 0
try:
    df = pd.read_excel(filepath, header=1) # UCI dataset usually has header on 2nd row (index 1)
    print("Columns:", df.columns.tolist())
    print("\nHead:\n", df.head())
    print("\nInfo:")
    print(df.info())
except Exception as e:
    print(e)
