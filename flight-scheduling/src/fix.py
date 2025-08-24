import pandas as pd

# Load your CSV
df = pd.read_csv('../data/Cleaned_Flight_Data.csv')

# Forward-fill Flight Number and Flight_Code columns
df['Flight Number'] = df['Flight Number'].ffill()
df['Flight_Code'] = df['Flight_Code'].ffill()

# Save back to CSV in the same format
df.to_csv('../data/Cleaned_Flight_Data.csv', index=False)

print("Flight Number and Flight_Code forward-filled and saved in Cleaned_Flight_Data.csv")