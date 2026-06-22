import pickle
import pandas as pd

with open('data/processed_model.pkl', 'rb') as f:
    assets = pickle.load(f)
    df = assets['books']
    print("Columns:", df.columns.tolist())
    print("Title type:", type(df['title']))
    print("Authors type:", type(df['authors']))
    print("First 5 titles:\n", df['title'].head())
