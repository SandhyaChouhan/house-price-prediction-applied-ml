import pandas as pd

# Load Excel dataset
df = pd.read_excel(r"D:\house_price_prediction\data\Case Study 1 Data.xlsx")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types & Nulls ---")
print(df.info())

print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())
