import requests
import pandas as pd

url = "https://docs.google.com/spreadsheets/d/1ypK_SACOonFlBM1MvWFqNLElwwSTUuinDHlSY5vBMjM/export?format=csv&gid=343215929"

# Download the CSV
response = requests.get(url)
response.raise_for_status()

# Save to file
with open('../data/listing_agents.csv', 'wb') as f:
    f.write(response.content)

print("Downloaded successfully!")

# Load and display basic info
df = pd.read_csv('../data/listing_agents.csv')
print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
