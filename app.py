import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from datetime import datetime
import random

# ==========================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="PoseGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for the Event Log
# This ensures our log doesn't clear out every time Streamlit re-runs
if 'event_log' not in st.session_state:
    st.session_state.event_log = pd.DataFrame(columns=["Timestamp", "Event", "Anomaly Probability"])

# ==========================================
# 2. SIDEBAR: CONTROLS & PRIVACY SETTINGS
# ==========================================
st.sidebar.title("🛡️ PoseGuard Controls")
st.sidebar.markdown("Privacy-Preserving Fall Detection System")
st.sidebar.divider()

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Test CSV Data", type=["csv"])

# Privacy Mode Toggle
privacy_mode = st.sidebar.radio(
    "Privacy Mode View",
    options=["Data Line Chart", "Abstract Stick Figure"],
    help="Toggle between raw sensor data trends and an abstracted skeletal view."
)

# Sensitivity Adjustment
# Lower Y-values mean the subject is closer to the floor.
threshold = st.sidebar.slider(
    "Fall Detection Sensitivity (Y-Axis Threshold)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="If the tracked Y-coordinate drops below this threshold, a fall is triggered."
)

st.sidebar.divider()

# Start/Stop Simulation Toggle
# Using a toggle is best for loops in Streamlit so the user can interrupt it seamlessly.
run_simulation = st.sidebar.toggle("▶️ Run Simulation", value=False)

# ==========================================
# 3. MAIN VIEW: REAL-TIME MONITORING
# ==========================================
st.title("Live Status Monitoring")

# Placeholders for dynamic content updating
alert_placeholder = st.empty()
chart_placeholder = st.empty()
st.subheader("Event Log")
log_placeholder = st.empty()

# Display the empty log initially
log_placeholder.dataframe(st.session_state.event_log, use_container_width=True, hide_index=True)

# ==========================================
# 4. SIMULATION LOOP LOGIC
# ==========================================
if run_simulation:
    if uploaded_file is None:
        st.sidebar.error("Please upload a CSV file to begin the simulation.")
    else:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        
        # Variables to maintain a sliding window for the Line Chart view
        window_size = 50
        history_y = []
        history_time = []
        
        # Iterate through the DataFrame to simulate real-time streaming
        for index, row in df.iterrows():
            # If the user turns off the toggle mid-simulation, break the loop
            if not st.session_state.run_simulation_toggle_state(run_simulation):
                # We check the widget state indirectly by relying on Streamlit's top-down rerun
                pass # The loop naturally halts when Streamlit reruns on toggle off
            
            # --- Detection Logic ---
            # Assuming 'head_y' is our primary tracking metric
            current_y = row['head_y']
            is_falling = current_y < threshold
            
            # --- Alert Trigger ---
            if is_falling:
                alert_placeholder.error(f"🚨 **URGENT: Fall Detected!** (Y-value dropped to {current_y:.2f})")
                
                # Calculate a mock anomaly probability based on how far past the threshold the value is
                prob = min(99.9, 80.0 + ((threshold - current_y) * 50))
                
                # Append to Event Log
                new_log = pd.DataFrame([{
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Event": "Fall Detected",
                    "Anomaly Probability": f"{prob:.2f}%"
                }])
                st.session_state.event_log = pd.concat([new_log, st.session_state.event_log]).head(10) # Keep last 10
            else:
                alert_placeholder.success("✅ Status: Normal - Monitoring active.")
                
            # Update the event log table
            log_placeholder.dataframe(st.session_state.event_log, use_container_width=True, hide_index=True)

            # --- Visualization Updates ---
            fig = go.Figure()

            if privacy_mode == "Data Line Chart":
                # Update sliding window
                history_y.append(current_y)
                history_time.append(index)
                if len(history_y) > window_size:
                    history_y.pop(0)
                    history_time.pop(0)
                
                # Plot the line chart
                fig.add_trace(go.Scatter(x=history_time, y=history_y, mode='lines+markers', name='Head Y-Coord', line=dict(color='#00ff00')))
                # Add the threshold line for visual reference
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Fall Threshold")
                
                fig.update_layout(
                    title="Real-Time Sensor Data (Y-Axis Tracking)",
                    xaxis_title="Frame / Time",
                    yaxis_title="Normalized Y-Coordinate",
                    yaxis_range=[0, 1.2],
                    template="plotly_dark", # Perfect for dark mode
                    height=400
                )

            elif privacy_mode == "Abstract Stick Figure":
                # Plot an abstract 3-point stick figure (Head -> Spine -> Feet)
                x_coords = [row['head_x'], row['spine_x'], row['feet_x']]
                y_coords = [row['head_y'], row['spine_y'], row['feet_y']]
                
                # Color turns red if falling, green if normal
                stick_color = 'red' if is_falling else '#00BFFF'
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords, 
                    mode='lines+markers',
                    marker=dict(size=[20, 10, 15], color=stick_color),
                    line=dict(width=6, color=stick_color)
                ))
                
                fig.update_layout(
                    title="Privacy-Preserving Skeletal Abstraction",
                    xaxis_range=[0, 1],
                    yaxis_range=[0, 1.2],
                    xaxis_visible=False, # Hide grid for a cleaner look
                    template="plotly_dark",
                    height=400
                )

            # Push the updated chart to the UI
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{index}")
            
            # Simulate real-time delay (e.g., 10 frames per second)
            time.sleep(0.1)

# Hack to read toggle state inside the loop
def get_toggle_state(var):
    return var
st.session_state.run_simulation_toggle_state = get_toggle_state