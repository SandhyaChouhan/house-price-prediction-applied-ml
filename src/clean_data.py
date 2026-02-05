import pandas as pd

# Load data
df = pd.read_excel("data/Case Study 1 Data.xlsx")

print("Initial shape:", df.shape)

# 1. Drop rows where Price is missing
df = df.dropna(subset=["Price"])

# 2. Fill numeric columns with median
numeric_cols = ["Size", "Bedrooms", "Bathrooms", "Year Built"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 3. Fill categorical column
df["Condition"] = df["Condition"].fillna("Unknown")

print("After cleaning shape:", df.shape)

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)

print("Cleaned data saved to data/cleaned_data.csv")
