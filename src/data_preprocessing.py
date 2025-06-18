import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['Type_of_order'] = df['Type_of_order'].str.strip()
    df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip()
    return df
