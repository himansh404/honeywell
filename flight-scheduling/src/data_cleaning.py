import pandas as pd
import numpy as np

def clean_and_prepare_data(excel_path, sheet_name, output_path):
    """
    A robust function to read the raw Excel data, clean it, create necessary
    features, and save it in a dashboard-ready format.
    """
    print("Starting data preparation...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # --- 1. Fill Core Identifiers ---
    # Forward-fill S.No and Flight Number for grouped data
    df['S.No'] = df['S.No'].ffill()
    df['Flight Number'] = df['Flight Number'].ffill()

    # --- 2. Standardize and Clean Columns ---
    # Rename columns for clarity and consistency
    if 'Unnamed: 2' in df.columns:
        df = df.rename(columns={'Unnamed: 2': 'Date'})
    df.columns = [col.replace(' ', '_') for col in df.columns] # Replace spaces in column names

    # Drop rows that are likely headers or empty
    df.dropna(subset=['Date', 'Flight_Number'], inplace=True)

    # Convert time columns to string to handle mixed types, then clean
    for col in ['STD', 'ATD', 'STA', 'ATA']:
        df[col] = df[col].astype(str).str.replace('-', '').str.strip()

    # --- 3. Create Airport and Direction Columns ---
    # This is the key logic to categorize each flight leg
    def assign_airport_and_direction(row):
        from_loc = str(row.get('From', '')).lower()
        to_loc = str(row.get('To', '')).lower()
        if 'mumbai' in from_loc: return 'Mumbai', 'Departure'
        if 'delhi' in from_loc: return 'Delhi', 'Departure'
        if 'mumbai' in to_loc: return 'Mumbai', 'Arrival'
        if 'delhi' in to_loc: return 'Delhi', 'Arrival'
        return None, None

    df[['Airport', 'Direction']] = df.apply(assign_airport_and_direction, axis=1, result_type='expand')
    df.dropna(subset=['Airport'], inplace=True) # Keep only flights related to our key airports

    # --- 4. Create Datetime and Delay Features ---
    df['Scheduled_Departure'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['STD'], errors='coerce')
    df['Actual_Departure'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['ATD'], errors='coerce')
    df['Scheduled_Arrival'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['STA'], errors='coerce')
    
    # Handle 'Landed HH:MM AM/PM' format in ATA
    landed_time = df['ATA'].str.extract(r'(\d{1,2}:\d{2}\s?[AP]M)')
    df['Actual_Arrival'] = pd.to_datetime(df['Date'].astype(str) + ' ' + landed_time[0].fillna(''), errors='coerce')

    df['Departure_Delay_Min'] = (df['Actual_Departure'] - df['Scheduled_Departure']).dt.total_seconds() / 60
    df['Arrival_Delay_Min'] = (df['Actual_Arrival'] - df['Scheduled_Arrival']).dt.total_seconds() / 60

    # --- 5. Final Selection and Save ---
    final_cols = [
        'Flight_Number', 'Airport', 'Direction', 'Date', 'From', 'To',
        'Scheduled_Departure', 'Actual_Departure', 'Scheduled_Arrival', 'Actual_Arrival',
        'Departure_Delay_Min', 'Arrival_Delay_Min'
    ]
    df_final = df[final_cols].copy()
    df_final.to_csv(output_path, index=False)
    print(f"Data preparation complete. Cleaned data saved to {output_path}")
    print("\nSample of prepared data:")
    print(df_final.head())

if __name__ == "__main__":
    clean_and_prepare_data(
        excel_path='../data/Flight_Data.xlsx',
        sheet_name='Sheet1',
        output_path='../data/Cleaned_Flight_Data.csv'
    )