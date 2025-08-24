import pandas as pd
import numpy as np

def generate_cascade_scenario():
    """
    Loads the clean flight data, creates an artificial scenario guaranteed to show
    a cascading delay, and saves it to a new CSV file.
    """
    try:
        # Define file paths
        original_data_path = '../data/Cleaned_Flight_Data.csv'
        dummy_data_path = '../data/Dummy_Cascade_Data.csv'

        # Load the original clean data
        df = pd.read_csv(
            original_data_path,
            parse_dates=['Date', 'Scheduled_Departure', 'Actual_Departure', 'Scheduled_Arrival', 'Actual_Arrival']
        )
        
        # --- Create the Artificial Scenario ---
        # We will focus on Delhi airport on a specific date
        target_airport = 'Delhi'
        target_date = '2025-08-22'
        
        # Find one arrival and one departure flight at the target airport and date
        arrivals = df[(df['Airport'] == target_airport) & (df['Date'].dt.strftime('%Y-%m-%d') == target_date) & (df['Direction'] == 'Arrival')]
        departures = df[(df['Airport'] == target_airport) & (df['Date'].dt.strftime('%Y-%m-%d') == target_date) & (df['Direction'] == 'Departure')]

        if arrivals.empty or departures.empty:
            print("Could not find suitable flights to create a dummy scenario. Please check the data.")
            return

        # Get the index of the first arrival and departure to modify
        arrival_idx = arrivals.index[0]
        departure_idx = departures.index[0]

        print(f"Modifying Arrival Flight: {df.loc[arrival_idx, 'Flight_Number']} and Departure Flight: {df.loc[departure_idx, 'Flight_Number']}")

        # 1. Create a tight scheduled turnaround
        # Set the departure to be 50 minutes after the arrival is scheduled to land.
        # Our model's minimum is 45 mins, so this is a tight but valid connection.
        original_arrival_time = df.loc[arrival_idx, 'Scheduled_Arrival']
        df.loc[departure_idx, 'Scheduled_Departure'] = original_arrival_time + pd.to_timedelta('50 minutes')

        # 2. Introduce a significant delay to the arriving flight
        # Let's say the arriving flight is 30 minutes late.
        arrival_delay_minutes = 30
        df.loc[arrival_idx, 'Arrival_Delay_Min'] = arrival_delay_minutes
        df.loc[arrival_idx, 'Actual_Arrival'] = df.loc[arrival_idx, 'Scheduled_Arrival'] + pd.to_timedelta(arrival_delay_minutes, 'm')

        # This setup means the 30-minute delay eats into the 50-minute turnaround,
        # leaving only 20 minutes. This is less than the 45-minute minimum,
        # which will trigger a cascading delay in our model.

        # Save the modified dataframe to a new file
        df.to_csv(dummy_data_path, index=False)
        print(f"Successfully created '{dummy_data_path}' with a cascade scenario.")

    except FileNotFoundError:
        print(f"Error: Could not find '{original_data_path}'. Please ensure the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_cascade_scenario()
