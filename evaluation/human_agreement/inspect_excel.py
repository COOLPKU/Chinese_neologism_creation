import pandas as pd
import os

file_path = r'd:\new_word\discriminate\final_data_1124\annotation\annotation_data_lql.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())
except Exception as e:
    print(f"Error reading excel: {e}")
