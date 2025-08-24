import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io

st.set_page_config(layout="wide", page_title="Flight-Ops AI Co-Pilot")

# --- Helper Functions ---
def format_hour(hour):
    if pd.isna(hour): return "N/A"
    hour = int(hour)
    return f"{hour % 12 or 12}:00 {'PM' if hour >= 12 else 'AM'}"

def get_color(delay):
    if pd.isna(delay): return 'grey'
    if delay > 45: return 'red'
    if delay > 20: return 'orange'
    return 'green'

# --- Backend Logic: Simple, Explainable, and Correct ---
def predict_delay_from_congestion(congestion):
    """A simple, explainable formula to predict delay."""
    BASE_DELAY = 10  # Base delay for any flight
    PENALTY_PER_FLIGHT = 5  # Extra 5 min delay for each flight in the same hour
    return BASE_DELAY + (congestion * PENALTY_PER_FLIGHT)

def run_single_flight_simulation(flight_index, current_schedule):
    """
    The 'Showcaser' backend. Simulates moving one flight to all possible hours
    to visualize the impact of congestion on delay.
    """
    flight_to_move = current_schedule.loc[flight_index]
    day_schedule = current_schedule[current_schedule['Date'] == flight_to_move['Date']].copy()
    day_schedule['Hour'] = day_schedule['Scheduled_Departure'].dt.hour
    
    hourly_congestion = day_schedule.groupby('Hour').size().to_dict()
    
    simulation_results = []
    for hour in range(24):
        # If we simulate moving to the original hour, the congestion is the original count.
        # If we simulate moving to a new hour, the congestion is the count of that new hour + our flight.
        if hour == flight_to_move['Scheduled_Departure'].hour:
            congestion = hourly_congestion.get(hour, 1)
        else:
            congestion = hourly_congestion.get(hour, 0) + 1
            
        predicted_delay = predict_delay_from_congestion(congestion)
        simulation_results.append({'Hour': hour, 'Predicted_Delay': predicted_delay, 'Congestion': congestion})
        
    return pd.DataFrame(simulation_results)

def run_full_day_optimizer(schedule_for_day):
    """
    The 'Optimizer' backend. Finds congested flights and moves them to the best available slots.
    This is the core of the solution.
    """
    if schedule_for_day.empty:
        return schedule_for_day, pd.DataFrame()

    df = schedule_for_day.copy()
    df['Hour'] = df['Scheduled_Departure'].dt.hour
    
    # 1. Identify Problem and Solution Hours
    hourly_congestion = df.groupby('Hour').size()
    congestion_threshold = hourly_congestion.quantile(0.75) # Top 25% most congested hours are "problems"
    
    problem_hours = hourly_congestion[hourly_congestion >= congestion_threshold].index.tolist()
    available_slots = hourly_congestion[hourly_congestion < congestion_threshold].sort_values()

    if not problem_hours or available_slots.empty:
        return schedule_for_day, pd.DataFrame() # No optimization possible

    # 2. Find flights to move
    flights_to_move = df[df['Hour'].isin(problem_hours)].sort_values('Departure_Delay_Min', ascending=False)
    
    changes_summary = []
    
    # 3. Systematically move flights
    for flight_index, flight_data in flights_to_move.iterrows():
        # Find the best available slot (least congested)
        if available_slots.empty:
            break # No more good slots
        
        best_new_hour = available_slots.index[0]
        
        # --- Record the change ---
        original_time = flight_data['Scheduled_Departure']
        original_delay = flight_data['Departure_Delay_Min']
        
        new_time = original_time.replace(hour=best_new_hour, minute=np.random.randint(0, 59)) # Add jitter
        new_congestion = available_slots.iloc[0] + 1
        new_delay = predict_delay_from_congestion(new_congestion)
        
        changes_summary.append({
            'Flight_Number': flight_data['Flight_Number'],
            'Original_Time': original_time.strftime('%H:%M'),
            'Original_Delay': f"{original_delay:.0f} min",
            'New_Time': new_time.strftime('%H:%M'),
            'New_Delay': f"{new_delay:.0f} min"
        })
        
        # --- Apply the change to the schedule ---
        df.loc[flight_index, 'Scheduled_Departure'] = new_time
        df.loc[flight_index, 'Departure_Delay_Min'] = new_delay
        df.loc[flight_index, 'Actual_Departure'] = new_time + pd.to_timedelta(new_delay, unit='m')
        
        # Update the congestion map
        available_slots.loc[best_new_hour] += 1
        available_slots.sort_values(inplace=True)

    return df, pd.DataFrame(changes_summary)

def identify_cascading_impact(schedule_for_day):
    """
    Identifies flights that have the biggest cascading impact on other flights
    due to tight turnaround times. This is our 'Cascade Impact' model.
    """
    MIN_TURNAROUND_MIN = 45  # Minimum required time to turn an aircraft around
    MAX_TURNAROUND_MIN = 120 # Maximum plausible time before it's likely a different plane

    df = schedule_for_day.copy().sort_values(by='Scheduled_Arrival')
    arrivals = df[df['Direction'] == 'Arrival'].to_dict('records')
    departures = df[df['Direction'] == 'Departure'].to_dict('records')

    impact_report = []

    for arr in arrivals:
        # For each arrival, find a plausible next departure for the same aircraft
        for dep in departures:
            # Check if the arrival's destination matches the departure's origin (implicit by airport filter)
            # and calculate the time between arrival and next departure
            turnaround_time = (dep['Scheduled_Departure'] - arr['Actual_Arrival']).total_seconds() / 60
            
            if MIN_TURNAROUND_MIN <= turnaround_time <= MAX_TURNAROUND_MIN:
                # This is a plausible link. Now, check for cascading delay.
                arrival_delay = arr['Arrival_Delay_Min']
                
                # The original scheduled time between flights
                scheduled_turnaround = (dep['Scheduled_Departure'] - arr['Scheduled_Arrival']).total_seconds() / 60
                
                # The cascade is the amount of time the arrival delay "eats into" the minimum required turnaround time.
                cascading_delay_caused = max(0, MIN_TURNAROUND_MIN - (scheduled_turnaround - arrival_delay))

                if cascading_delay_caused > 0:
                    impact_report.append({
                        'Arriving Flight': arr['Flight_Number'],
                        'Origin': arr['Origin'],
                        'Arrival Delay (min)': f"{arrival_delay:.0f}",
                        'Linked Outbound Flight': dep['Flight_Number'],
                        'Destination': dep['Destination'],
                        'Scheduled Turnaround (min)': f"{scheduled_turnaround:.0f}",
                        'Cascading Delay Caused (min)': f"{cascading_delay_caused:.0f}",
                    })
                # Once a link is found, break to avoid linking one arrival to multiple departures
                break 
    
    if not impact_report:
        return pd.DataFrame()

    report_df = pd.DataFrame(impact_report)
    # Convert to numeric for sorting
    report_df['Cascading Delay Caused (min)'] = pd.to_numeric(report_df['Cascading Delay Caused (min)'])
    return report_df.sort_values(by='Cascading Delay Caused (min)', ascending=False).head(10)

# --- Data Loading and State Management ---
@st.cache_data
def load_data():
    try:
        # Use absolute path from repo root
        data = pd.read_csv('flight-scheduling/data/Cleaned_Flight_Data.csv', parse_dates=['Date', 'Scheduled_Departure', 'Actual_Departure', 'Scheduled_Arrival', 'Actual_Arrival'])
        return data
    except FileNotFoundError:
        return None

df = load_data()
if df is None:
    st.error("`Cleaned_Flight_Data.csv` not found. Please run `data_cleaning.py` first.")
    st.stop()

if 'tuned_df' not in st.session_state:
    st.session_state.tuned_df = df.copy()
    st.session_state.tuned_df['unique_id'] = range(len(st.session_state.tuned_df))
    st.session_state.tuned_df.set_index('unique_id', inplace=True)



def get_color(delay):
    if pd.isna(delay): return 'grey'
    if delay > 45: return 'red'
    if delay > 20: return 'orange'
    return 'green'

def process_nlp_query(query, df, airport):
    """
    A simple NLP processor to answer questions about flight data for a selected airport.
    """
    query = query.lower()
    df_airport = df[df['Airport'] == airport]

    # Query 1: Average delay
    if "average delay" in query:
        avg_dep_delay = df_airport[df_airport['Direction'] == 'Departure']['Departure_Delay_Min'].mean()
        avg_arr_delay = df_airport[df_airport['Direction'] == 'Arrival']['Arrival_Delay_Min'].mean()
        return f"For {airport}, the average departure delay is {avg_dep_delay:.0f} minutes and the average arrival delay is {avg_arr_delay:.0f} minutes."

    # Query 2: Busiest hour
    if "busiest hour" in query or "most congested" in query:
        busiest_hour = df_airport['Scheduled_Departure'].dt.hour.value_counts().idxmax()
        return f"The busiest hour for departures at {airport} is {format_hour(busiest_hour)}."

    # Query 3: Status of a specific flight
    if "flight status" in query or "status of flight" in query:
        # Try to find a flight number in the query
        flight_number_match = [word.upper() for word in query.split() if '-' in word or any(char.isdigit() for char in word)]
        if not flight_number_match:
            return "Please specify a flight number (e.g., 'status of flight 6E-237')."
        
        flight_number = flight_number_match[0]
        flight_data = df_airport[df_airport['Flight_Number'] == flight_number]

        if flight_data.empty:
            return f"Sorry, I could not find any data for flight {flight_number} at {airport}."
        
        # For simplicity, report the first instance found
        flight = flight_data.iloc[0]
        direction = flight['Direction']
        delay = flight[f'{direction}_Delay_Min']
        status = "On Time" if delay <= 0 else f"Delayed by {delay:.0f} minutes"
        return f"Flight {flight_number} ({direction}) at {airport} is currently reported as: {status}."    
    # Default response
    return "Sorry, I can only answer questions about 'average delay', 'busiest hour', or 'flight status [flight number]'. Please try one of those."


# --- Main UI ---
st.title("✈️ Flight-Ops AI Co-Pilot")
st.markdown("An AI-powered dashboard to de-congest traffic and optimize flight schedules for busy airports.")

selected_airport = st.selectbox('Select an Airport to Analyze', options=sorted(df['Airport'].unique()), key="main_airport_selector")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Dashboard", "🔧 Optimizer & Simulator", "💥 Cascade Impact Analysis", "💬 NLP & Download"])

# --- Tab 1: Performance Dashboard ---
with tab1:
    st.header(f"Performance Dashboard for {selected_airport}")
    df_filtered = st.session_state.tuned_df[st.session_state.tuned_df['Airport'] == selected_airport].copy()

    for direction in ['Departure', 'Arrival']:
        st.subheader(f"Analysis for {direction}s")
        df_dir = df_filtered[df_filtered['Direction'] == direction]
        if df_dir.empty:
            st.warning(f"No {direction} data available.")
            continue

        hour_col = f'{direction}_Hour'
        delay_col = f'{direction}_Delay_Min'
        df_dir.loc[:, hour_col] = df_dir[f'Scheduled_{direction}'].dt.hour

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Busiest Hours (Congestion)")
            hourly_counts = df_dir[hour_col].value_counts().sort_index()
            fig_counts = px.bar(hourly_counts, x=hourly_counts.index, y=hourly_counts.values, labels={'x': 'Hour of Day', 'y': 'Number of Flights'})
            st.plotly_chart(fig_counts, use_container_width=True)
        with col2:
            st.markdown("#### Average Delay by Hour")
            hourly_delays = df_dir.groupby(hour_col)[delay_col].mean().sort_index()
            hourly_delays_df = hourly_delays.reset_index()
            hourly_delays_df['Color'] = hourly_delays_df[delay_col].apply(get_color)
            fig_delays = px.bar(hourly_delays_df, x=hour_col, y=delay_col, color='Color',
                                color_discrete_map={"green": "#2ca02c", "orange": "#ff7f0e", "red": "#d62728", "grey": "#808080"},
                                labels={'x': 'Hour of Day', 'y': 'Average Delay (minutes)'})
            st.plotly_chart(fig_delays, use_container_width=True)

# --- Tab 2: Optimizer & Simulator ---
with tab2:
    st.header("🔧 Full Day Optimizer & Simulator")
    st.markdown("Use the **Optimizer** to fix an entire day's schedule, or use the **Simulator** to analyze a single flight's delay sensitivity.")

    date_options = sorted(df['Date'].dt.strftime('%Y-%m-%d').unique().tolist())
    selected_date_str = st.selectbox("1. Select a Date to Work With", options=date_options, key="optimizer_date_selector")

    st.subheader("🚀 Full Day Schedule Optimizer")
    st.markdown("This tool automatically reschedules flights from congested hours to less busy slots to reduce overall delays.")
    
    if st.button("Optimize Full Day's Schedule", type="primary"):
        schedule_to_optimize = st.session_state.tuned_df[
            (st.session_state.tuned_df['Airport'] == selected_airport) &
            (st.session_state.tuned_df['Date'].dt.strftime('%Y-%m-%d') == selected_date_str) &
            (st.session_state.tuned_df['Direction'] == 'Departure')
        ].copy()
        
        optimized_schedule, changes_summary = run_full_day_optimizer(schedule_to_optimize)
        
        if not changes_summary.empty:
            st.session_state.tuned_df.update(optimized_schedule)
            st.subheader("✅ Optimization Complete: Summary of Changes")
            st.dataframe(changes_summary, use_container_width=True)
            st.success(f"Successfully rescheduled {len(changes_summary)} flights. View the updated schedule on the 'Performance Dashboard'.")
        else:
            st.warning("No optimizations were necessary or possible for the selected day.")

    st.divider()

    st.subheader("🔍 Single Flight Simulator")
    st.markdown("Select a single flight to visualize how its predicted delay changes based on hourly congestion. This is for analysis only and does not change the schedule.")
    
    df_sim = st.session_state.tuned_df[
        (st.session_state.tuned_df['Airport'] == selected_airport) &
        (st.session_state.tuned_df['Date'].dt.strftime('%Y-%m-%d') == selected_date_str) &
        (st.session_state.tuned_df['Direction'] == 'Departure')
    ].copy()

    if df_sim.empty:
        st.warning("No departure data available for simulation on the selected date.")
    else:
        flight_options = df_sim.sort_values('Scheduled_Departure')['Flight_Number'].unique()
        selected_flight_number = st.selectbox("Select Flight Number to Simulate", options=flight_options)
        
        flight_instances = df_sim[df_sim['Flight_Number'] == selected_flight_number]
        if not flight_instances.empty:
            # To handle multiple flights with the same number on the same day, let user pick one
            flight_instances['display'] = flight_instances['Scheduled_Departure'].dt.strftime('%Y-%m-%d %H:%M')
            selected_instance_display = st.selectbox("Select Specific Flight Instance", options=flight_instances['display'])
            selected_flight_index = flight_instances[flight_instances['display'] == selected_instance_display].index[0]

            st.markdown(f"**Simulating Delay for Flight {selected_flight_number}**")
            simulation_df = run_single_flight_simulation(selected_flight_index, st.session_state.tuned_df)
            
            fig = px.bar(simulation_df, x='Hour', y='Predicted_Delay',
                         hover_data=['Congestion'],
                         labels={'Hour': 'Scheduled Departure Hour', 'Predicted_Delay': 'Predicted Delay (minutes)'},
                         title=f"Predicted Delay vs. Departure Hour")
            st.plotly_chart(fig, use_container_width=True)
            st.info("This chart demonstrates the core principle: predicted delay is lowest during hours with the least congestion.")
        else:
            st.warning(f"Flight {selected_flight_number} has no instances on the selected date.")

# --- Tab 3: Cascade Impact Analysis ---
with tab3:
    st.header("💥 Cascade Impact Analysis")
    st.markdown("This tool identifies 'super-spreader' flights. These are flights whose arrival delays are most likely to cause a cascading delay for their next outbound flight due to tight turnaround times.")
    st.info("💡 **How it works:** We infer aircraft links by finding an arrival and a subsequent departure from the same airport within a realistic turnaround window (45-120 min). We then calculate how much an arrival delay eats into the minimum required turnaround time.")

    date_options_cascade = sorted(df['Date'].dt.strftime('%Y-%m-%d').unique().tolist())
    selected_date_cascade = st.selectbox("1. Select a Date to Analyze", options=date_options_cascade, key="cascade_date_selector")

    if st.button("Analyze Cascading Delays", type="primary"):
        schedule_to_analyze = st.session_state.tuned_df[
            (st.session_state.tuned_df['Airport'] == selected_airport) &
            (st.session_state.tuned_df['Date'].dt.strftime('%Y-%m-%d') == selected_date_cascade)
        ].copy()

        if schedule_to_analyze.empty:
            st.warning("No flight data for the selected airport and date.")
        else:
            with st.spinner("Analyzing flight links and calculating impact..."):
                impact_df = identify_cascading_impact(schedule_to_analyze)
            
            if not impact_df.empty:
                st.subheader("Top 10 Flights Causing Cascading Delays")
                st.dataframe(impact_df, use_container_width=True, hide_index=True)
                st.success("Analysis complete. The table above shows the flights most likely to cause downstream delays.")
            else:
                st.info("No significant cascading delays were identified for this day. This could be due to sufficient buffer times in the schedule.")

# --- Tab 4: NLP & Download ---
with tab4:
    st.header("💬 NLP Chatbot")
    st.markdown(f"Ask me questions about **{selected_airport}**. For example:")
    st.code("- What is the average delay?\n- What is the busiest hour?\n- What is the status of flight 6E-237?")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input(f"Your question about {selected_airport}..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            response = process_nlp_query(prompt, st.session_state.tuned_df, selected_airport)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    st.divider()
    st.header("Download Your Optimized Schedule")
    st.markdown("Download the CSV with all the changes you have applied using the Optimizer.")
    
    @st.cache_data
    def to_csv(df_to_download):
        download_df = df_to_download.copy()
        if 'unique_id' in download_df.columns or download_df.index.name == 'unique_id':
             download_df = download_df.reset_index(drop=True)
        output = io.BytesIO()
        download_df.to_csv(output, index=False, encoding='utf-8')
        return output.getvalue()

    tuned_csv = to_csv(st.session_state.tuned_df)
    st.download_button(label="📥 Download Tuned Schedule (.csv)", data=tuned_csv, file_name="tuned_schedule.csv", mime="text/csv")
