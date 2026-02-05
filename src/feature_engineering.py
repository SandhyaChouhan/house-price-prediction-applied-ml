import pandas as pd

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Convert Date Sold to datetime
df["Date Sold"] = pd.to_datetime(df["Date Sold"])

# Extract time features
df["Sold_Year"] = df["Date Sold"].dt.year
df["Sold_Month"] = df["Date Sold"].dt.month

# Create Property Age
df["Property_Age"] = df["Sold_Year"] - df["Year Built"]

# Drop unnecessary columns
df = df.drop(columns=["Property ID", "Date Sold"])

print("Feature engineering completed.")
print(df.head())

# Save final dataset
df.to_csv("data/final_data.csv", index=False)

print("Final dataset saved to data/final_data.csv")
