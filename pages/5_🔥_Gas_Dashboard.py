import streamlit as st
st.set_page_config(
    page_title="Hotels Gas Dashboard",  
    page_icon="🔥",       
    layout="wide",        
    initial_sidebar_state="expanded",  
    menu_items={
        'Get Help': 'https://www.streamlit.io/community',
        'Report a bug': "https://github.com/yourusername/yourrepo/issues",
        'About': "# This is a dashboard for monitoring hotel gas usage."
    }
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import calendar
import os
import numpy as np
from datetime import datetime, timedelta
import openai
import sqlite3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import base64
import io
import time
# Add these imports at the top
import plotly.io as pio
try:
    import kaleido
except ImportError:
    st.warning("Kaleido package not found. Chart export in emails may not work.")
    st.info("Install with: pip install kaleido")
import gspread
from google.oauth2.service_account import Credentials
from google.oauth2 import service_account
from datetime import datetime
from datetime import date as dt_date  # Add this at the top if not already imported



# Define color scheme
app_colors = {
    "background": "#f3f4f6",   # Light gray background
    "header": "#3A3F87",       # Deep blue-violet for headers
    "text": "#2C2E3E",         # Dark charcoal text
    "card_bg": "#ffffff",      # White for cards
    "dropdown_bg": "#f5f7fa",  # Light gray for dropdowns
    "accent": "#FF8C00",       # Orange accent for gas (different from electricity blue)
    "highlight": "#FF4D6D"     # Bold pink-red for highlights
}

# Location to Hotel mapping (based on your provided mapping)
location_to_hotel = {
    "28-30 Jamestown Road - Camden - Camden": "Camden",
    "Energy Centre - Aldgate - Canopy": "Canopy",
    "18-20 Belgrave Road - CIV": "CIV",
    "36-40 Belgrave Road - EH": "EH",
    "Holiday Inn Express St Albans - Colney Fields Shopping Park - Barnet Road - St Albans": "St Albans"
}

# Add this dictionary for gas carbon factors
GAS_FACTORS = {
    "2025": 0.18150,
    "2024": 0.18290,
    "2023": 0.18293,
    "2022":  0.18254

}

# Gas price per kWh
GAS_PRICE = 0.05  # £/kWh


@st.cache_data(show_spinner=False)
def load_data():
    try:
        conn = sqlite3.connect('data/energy_data.db')
        
        # First check what's actually in the database
        check_query = pd.read_sql_query("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND (name='gas_hh_data' OR name='gas_hh_pivot_data')
        """, conn)
        
        if check_query.empty:
            st.error("Gas data tables do not exist in the database. Please run the gas data processor first.")
            return None
        
        # For the pivot table with half-hourly data
        data = pd.read_sql_query("""
        SELECT * FROM gas_hh_pivot_data
        ORDER BY Date
        """, conn)
        
        # Convert Date column to datetime explicitly
        data['Date'] = pd.to_datetime(data['Date'], format='mixed')
        
        # Map locations to hotel names
        data["Hotel"] = data["Location"].map(location_to_hotel)
        
        # Get time columns - those containing a colon
        time_cols = [col for col in data.columns if ':' in col]
        
        # Ensure numeric values
        for col in time_cols + ['Total Usage']:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Add time columns
        data["Year"] = data["Date"].dt.year
        data["Month"] = data["Date"].dt.month
        data["Day of Week"] = data["Date"].dt.day_name()
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()
def check_for_duplicates(data):
    """
    Check for and handle duplicate data entries.
    Returns the cleaned data and information about duplicates.
    """
    # Count total rows before deduplication
    total_rows = len(data)
    
    # Find duplicate dates per hotel
    dupes = data.duplicated(subset=['Date', 'Hotel'], keep=False)
    duplicate_rows = data[dupes].copy()
    
    # Count the number of duplicated rows
    duplicate_count = len(duplicate_rows)
    
    if duplicate_count > 0:
        # st.warning(f"⚠️ Found {duplicate_count} duplicate entries in the data.")
        
        # # You can display some examples of duplicates
        # with st.expander("Show examples of duplicate entries"):
        #     st.dataframe(duplicate_rows.head(10))
        
        # Examine which hotels have duplicates
        duplicate_hotels = duplicate_rows['Hotel'].unique()
        duplicate_dates = duplicate_rows['Date'].dt.date.unique()
        
        # st.info(f"Hotels with duplicates: {', '.join(duplicate_hotels)}")
        # st.info(f"Number of dates with duplicates: {len(duplicate_dates)}")
        
        # Option 1: Keep the first occurrence of each date/hotel combination
        data_deduped = data.drop_duplicates(subset=['Date', 'Hotel'], keep='first')
        
        # st.success(f"Removed {total_rows - len(data_deduped)} duplicate rows.")
        return data_deduped
    else:
        return data  # No duplicates found
    
def safe_year_replace(date, new_year):
    try:
        return date.replace(year=new_year)
    except ValueError:
        # If February 29 in leap year, return February 28 in non-leap year
        return date.replace(year=new_year, day=28)

def create_heatmap(data, selected_hotel, selected_year, selected_month):
    """Create enhanced heatmap from half-hourly data"""
    # Filter data
    mask = (
        (data["Hotel"] == selected_hotel) & 
        (data["Year"] == selected_year) & 
        (data["Month"] == selected_month)
    )
    filtered_data = data.loc[mask].copy()
    
    # Get half-hourly columns
    time_cols = [col for col in data.columns if ':' in col]
    
    # Ensure numeric values
    for col in time_cols:
        filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
    
    # Add day of week if not present
    if "Day of Week" not in filtered_data.columns:
        filtered_data["Day of Week"] = filtered_data["Date"].dt.day_name()
    
    # Calculate average usage for each time period and day
    pivot_data = pd.DataFrame()
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for day in days_order:
        day_data = filtered_data[filtered_data["Day of Week"] == day]
        if not day_data.empty:
            pivot_data[day] = day_data[time_cols].mean()
    
    # Transpose to get days as rows and times as columns
    pivot_data = pivot_data.T
    
    # Create heatmap with an orange colorscale for gas
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=time_cols,
        y=days_order,
        colorscale="Oranges",  # Change to Oranges for gas
        colorbar=dict(title="Usage (kWh)"),
        hoverongaps=False,
        hovertemplate="Day: %{y}<br>Time: %{x}<br>Usage: %{z:.2f} kWh<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Average Half-Hourly Gas Usage by Day - {calendar.month_name[selected_month]} {selected_year}",
        xaxis_title="Time of Day",
        yaxis_title="Day of Week",
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            ticktext=[f"{hour:02d}:00" for hour in range(24)],
            tickvals=[f"{hour:02d}:00" for hour in range(24)]
        ),
        height=500
    )
    
    return fig

def get_hotel_facilities(selected_hotel):
    """Define hotel facilities based on hotel type"""
    hotel_types = {
        "CIER": {
            "category": "Midscale",
            "has_pool": False,
            "has_restaurant": True,
            "has_conf_rooms": False,
            "room_count": 38
        },
        "CIV": {
            "category": "Midscale",
            "has_pool": False,
            "has_restaurant": True,
            "has_conf_rooms": False,
            "room_count": 50
        },
        "Westin": {
            "category": "Luxury",
            "has_pool": True,
            "has_restaurant": True,
            "has_conf_rooms": True,
            "room_count": 220
        },
        "Canopy": {
            "category": "Upscale",
            "has_pool": False,
            "has_restaurant": True,
            "has_conf_rooms": True,
            "room_count": 340
        },
        "Camden": {
            "category": "Midscale",
            "has_pool": False,
            "has_restaurant": True,
            "has_conf_rooms": True,
            "room_count": 130
        },
        "EH": {
            "category": "Economy",
            "has_pool": False,
            "has_restaurant": False,
            "has_conf_rooms": False,
            "room_count": 105
        },
        "St Albans": {
            "category": "Economy",
            "has_pool": False,
            "has_restaurant": True,
            "has_conf_rooms": True,
            "room_count": 96
        }
    }
    return hotel_types.get(selected_hotel, {})

def calculate_baseline(current_month_data, last_year_data, time_cols, selected_date):
    """Calculate baseline using current month and last year's data"""
    selected_day = pd.to_datetime(selected_date)
    is_weekend = selected_day.weekday() >= 5
    
    # Get corresponding date from last year
    last_year_date = selected_day - pd.DateOffset(years=1)
    
    baseline_usage = {}
    for time in time_cols:
        if time in current_month_data.columns:
            # Current month similar days (weekday/weekend)
            current_similar_days = current_month_data[
                (current_month_data['Date'].dt.weekday.ge(5) == is_weekend) & 
                (current_month_data['Date'] != selected_date)
            ]
            current_avg = current_similar_days[time].mean()
            
            # Last year similar days
            last_year_similar_days = last_year_data[
                last_year_data['Date'].dt.weekday.ge(5) == is_weekend
            ]
            last_year_avg = last_year_similar_days[time].mean()
            
            # Weighted combination (adjust weights based on data quality)
            if pd.notna(current_avg) and pd.notna(last_year_avg):
                baseline_usage[time] = (current_avg * 0.6) + (last_year_avg * 0.4)
            elif pd.notna(current_avg):
                baseline_usage[time] = current_avg
            elif pd.notna(last_year_avg):
                baseline_usage[time] = last_year_avg
            else:
                baseline_usage[time] = None
    
    return baseline_usage

def show_overview_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month, hotel_facilities, occupancy_rate):
    """Overview tab with current vs last year comparison"""
    
    st.header("📊 Property Overview")
    
    # Calculate metrics based on days in current month
    days_in_month = len(monthly_data['Date'].dt.date.unique())  # Count unique dates
    total_usage = monthly_data["Total Usage"].sum()
    
    # Get last year's data for the same period
    last_year_mask = (
        (hotel_data["Year"] == selected_year - 1) & 
        (hotel_data["Month"] == selected_month) &
        (hotel_data["Date"].dt.day <= days_in_month)  # Same number of days
    )
    last_year_usage = hotel_data.loc[last_year_mask, "Total Usage"].sum()
    
    # Project full month
  
    projected_usage = (total_usage / days_in_month) * calendar.monthrange(selected_year, selected_month)[1]
    
    rooms = hotel_facilities.get('room_count', 100)
    occupied_rooms = rooms * (occupancy_rate/100)
    
    # Key metrics in 2x2 grid
    col1, col2 = st.columns(2)
    with col1:
        # Current usage vs last year (same days)
        if last_year_usage > 0:
            ytd_change = ((total_usage - last_year_usage) / last_year_usage) * 100
            st.metric(
                "Monthly Gas Usage",
                f"{total_usage:,.0f} kWh",
                f"{ytd_change:+.1f}% vs Last Year (same {days_in_month} days)",
                delta_color="inverse",
                help=f"Comparison with same period last year"
            )
        else:
            st.metric(
                "Monthly Gas Usage", 
                f"{total_usage:,.0f} kWh",
                help=f"Gas usage for first {days_in_month} days"
            )
                    
        # Projected monthly consumption
        st.metric(
            "Projected Full Month",
            f"{projected_usage:,.0f} kWh",
            help=f"Projected based on current daily average"
        )
    
    with col2:
        # Daily average comparison
        current_daily_avg = total_usage / days_in_month
        last_year_daily_avg = last_year_usage / days_in_month if days_in_month > 0 else 0
        
        if last_year_daily_avg > 0:
            daily_avg_change = ((current_daily_avg - last_year_daily_avg) / last_year_daily_avg) * 100
            st.metric(
                "Daily Average",
                f"{current_daily_avg:,.1f} kWh",
                f"{daily_avg_change:+.1f}% vs Last Year",
                delta_color="inverse",
                help="Average daily consumption comparison"
            )
        else:
            st.metric(
                "Daily Average",
                f"{current_daily_avg:,.1f} kWh",
                help="Current average daily gas consumption"
            )
    
    # Daily comparison visualization
    st.subheader("Daily Gas Usage Comparison")
    
    # Get daily data for current and last year
    current_daily = monthly_data.groupby("Date")["Total Usage"].sum().reset_index()
    
    last_year_daily = hotel_data[
        (hotel_data["Year"] == selected_year - 1) & 
        (hotel_data["Month"] == selected_month)
    ].groupby("Date")["Total Usage"].sum().reset_index()
    
    # Use safe date conversion
    last_year_daily["Date"] = last_year_daily["Date"].apply(
        lambda x: safe_year_replace(x, selected_year)
    )
    
    fig = go.Figure()
    
    # Current year line (orange for gas)
    fig.add_trace(go.Scatter(
        x=current_daily["Date"],
        y=current_daily["Total Usage"],
        mode='lines+markers',
        name=str(selected_year),
        line=dict(color='#FF8C00', width=2),  # Orange color for gas
        marker=dict(size=6)
    ))
    
    # Last year line
    fig.add_trace(go.Scatter(
        x=last_year_daily["Date"],
        y=last_year_daily["Total Usage"],
        mode='lines+markers',
        name=str(selected_year - 1),
        line=dict(color='#FF4D6D', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Weekend highlighting
    for date in current_daily["Date"]:
        if date.weekday() >= 5:  # Weekend
            fig.add_vrect(
                x0=date,
                x1=date + pd.Timedelta(days=1),
                fillcolor="rgba(128, 128, 128, 0.1)",
                layer="below",
                line_width=0
            )
    dates_with_notes = get_dates_with_notes(selected_hotel, selected_year, selected_month)
    if dates_with_notes:
        # Convert to datetime for compatibility with Plotly
        note_dates = [pd.Timestamp(d) for d in dates_with_notes]
        
        # Get y-values for these dates from current_daily
        y_values = []
        tooltips = []
        
        for note_date in note_dates:
            # Find the corresponding data point in current_daily
            matching_rows = current_daily[current_daily['Date'].dt.date == note_date.date()]
            
            if not matching_rows.empty:
                y_value = matching_rows['Total Usage'].iloc[0]
                y_values.append(y_value)
                
                # Get the first note for this date to show in tooltip
                date_notes = get_notes(selected_hotel, date=note_date)
                if date_notes:
                    first_note = date_notes[0]['note_text']
                    # Truncate long notes for tooltip
                    if len(first_note) > 50:
                        first_note = first_note[:47] + "..."
                    tooltips.append(f"Note: {first_note}")
                else:
                    tooltips.append("Has notes")
            else:
                # Skip dates without data
                continue
        
        if y_values:  # Only add markers if we have valid dates with data
            fig.add_trace(go.Scatter(
                x=[note_date for note_date in note_dates if note_date.date() in current_daily['Date'].dt.date.values],
                y=y_values,
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=12,
                    color='#FFD700',  # Gold color for the stars
                    line=dict(width=1, color='black')
                ),
                name='Has Notes',
                text=tooltips,
                hoverinfo='text'
            ))
    # END OF ADDED CODE BLOCK
            
    
    fig.update_layout(
        title=f"Daily Gas Usage: {calendar.month_name[selected_month]} {selected_year} vs {selected_year-1} (Weekends Shaded)",
        xaxis_title="Date",
        yaxis_title="Gas Usage (kWh)",
        hovermode="x unified",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def process_anomalies(day_data, time_cols, baseline_usage):
    """Process and identify anomalies in daily data"""
    day_data_long = day_data.melt(
        id_vars=['Date'], 
        value_vars=time_cols, 
        var_name='Time', 
        value_name='Usage'
    )
    
    day_data_long['Baseline'] = day_data_long['Time'].map(baseline_usage)
    
    # Safe division for deviation calculation
    day_data_long['Deviation'] = np.where(
        day_data_long['Baseline'] != 0,
        ((day_data_long['Usage'] - day_data_long['Baseline']) / day_data_long['Baseline'] * 100),
        0
    )
    
    day_data_long['Anomaly'] = day_data_long['Deviation'].apply(
        lambda x: "High Anomaly" if x > 20 else "Low Anomaly" if x < -20 else None
    )
    
    return day_data_long

def create_anomaly_visualization(day_data_long, selected_date):
    """Create visualization for anomaly detection"""
    fig = go.Figure()
    
    # Add actual usage line
    fig.add_trace(go.Scatter(
        x=day_data_long['Time'],
        y=day_data_long['Usage'],
        name='Actual Usage',
        line=dict(color='#FF8C00', width=2)  # Orange for gas
    ))
    
    # Add baseline
    fig.add_trace(go.Scatter(
        x=day_data_long['Time'],
        y=day_data_long['Baseline'],
        name='Baseline',
        line=dict(color='#808080', dash='dash')
    ))
    
    # Add anomaly points
    high_anomalies = day_data_long[day_data_long['Anomaly'] == "High Anomaly"]
    low_anomalies = day_data_long[day_data_long['Anomaly'] == "Low Anomaly"]
    
    if not high_anomalies.empty:
        fig.add_trace(go.Scatter(
            x=high_anomalies['Time'],
            y=high_anomalies['Usage'],
            mode='markers',
            name='High Anomaly',
            marker=dict(color='red', size=10)
        ))
    
    if not low_anomalies.empty:
        fig.add_trace(go.Scatter(
            x=low_anomalies['Time'],
            y=low_anomalies['Usage'],
            mode='markers',
            name='Low Anomaly',
            marker=dict(color='blue', size=10)
        ))
    
    # Get day of week
    day_of_week = pd.to_datetime(selected_date).strftime('%A')
    
    fig.update_layout(
        title=f"Gas Usage Pattern Analysis - {day_of_week}, {selected_date}",
        xaxis_title="Time",
        yaxis_title="Gas Usage (kWh)",
        hovermode="x unified",
        showlegend=True
    )
    
    return fig


def show_realtime_tab(monthly_data, hotel_data, hotel_facilities, occupancy_rate, selected_year, selected_month):
    """Real-time tab focused on spotting anomalies in energy usage patterns"""
    st.header("🔥 Gas Usage Anomaly Detection")
    
    # Check if we have data
    if monthly_data.empty:
        st.warning("No data available for the selected period.")
        return
        
    # Add baseline explanation
    with st.expander("ℹ️ How is baseload calculated?"):
        st.markdown("""
        ### Baseload Analysis
        
        For gas consumption, baseload is calculated as:
        - **Overnight and Weekend Usage**: The average consumption during nighttime hours (1-5 AM)
        - **Minimum Load**: The lowest consistent usage pattern when occupancy and activity are minimal
        
        This approach helps identify:
        - Equipment that's running unnecessarily
        - Heating systems that aren't set back properly
        - Potential gas leaks or inefficiencies
        
        Anomalies are flagged when usage significantly exceeds expected baseload during low-activity periods.
        """)
    
    # Get selected hotel from monthly data
    selected_hotel = monthly_data["Hotel"].iloc[0] if not monthly_data.empty else None
    
    if selected_hotel is None:
        st.warning("No hotel selected or no data available.")
        return
    
    # Get time columns
    time_cols = [col for col in monthly_data.columns if ':' in col]
    
    # Calculate baseload (overnight hours)
    overnight_cols = [col for col in time_cols if col.startswith(('01:', '02:', '03:', '04:', '05:'))]
    
    if not overnight_cols:
        st.error("Unable to calculate baseload - missing overnight data.")
        return
        
    # Calculate baseload for each day
    baseload_data = monthly_data[overnight_cols].mean(axis=1).reset_index()
    baseload_data.columns = ['index', 'Baseload']
    
    # Calculate average baseload
    avg_baseload = baseload_data['Baseload'].mean()
    
    # Date selection with validation
    st.subheader("Select Date for Analysis")
    available_dates = monthly_data['Date'].dt.date.unique()
    
    if len(available_dates) > 0:
        selected_date = st.date_input(
            "Choose a day",
            value=available_dates[0],
            min_value=min(available_dates),
            max_value=max(available_dates)
        )
        
        # Check if data exists for selected date
        selected_day_data = monthly_data[monthly_data['Date'].dt.date == selected_date]
        
        if selected_day_data.empty:
            st.error(f"No data available for {selected_date}. Please select another date.")
            return
        
        # Get baseload for the selected day
        day_baseload = selected_day_data[overnight_cols].mean(axis=1).iloc[0]
        
        # Create long-format data for visualization
        day_data_long = selected_day_data.melt(
            id_vars=['Date'], 
            value_vars=time_cols, 
            var_name='Time', 
            value_name='Usage'
        )
        
        # Add baseload reference
        day_data_long['Baseload'] = day_baseload
        
        # Calculate deviation from baseload
        day_data_long['Deviation'] = ((day_data_long['Usage'] - day_data_long['Baseload']) / day_data_long['Baseload'] * 100)
        
        # Identify anomalies - high usage during expected low-usage periods
        day_data_long['Anomaly'] = None
        
        # Overnight hours (1-5 AM) should be close to baseload
        overnight_mask = day_data_long['Time'].apply(lambda x: x.startswith(('01:', '02:', '03:', '04:', '05:')))
        day_data_long.loc[overnight_mask & (day_data_long['Deviation'] > 50), 'Anomaly'] = "High Overnight Usage"
        
        # Weekend high usage (if applicable)
        is_weekend = pd.to_datetime(selected_date).weekday() >= 5
        if is_weekend:
            # Higher threshold for weekends
            day_data_long.loc[day_data_long['Deviation'] > 100, 'Anomaly'] = "High Weekend Usage"
        
        # Display anomaly metrics
        high_anomalies = day_data_long[day_data_long['Anomaly'] == "High Overnight Usage"]
        weekend_anomalies = day_data_long[day_data_long['Anomaly'] == "High Weekend Usage"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Day Baseload", f"{day_baseload:.2f} kWh")
        with col2:
            st.metric("Average Baseload", f"{avg_baseload:.2f} kWh")
        with col3:
            baseload_diff = ((day_baseload - avg_baseload) / avg_baseload * 100) if avg_baseload > 0 else 0
            st.metric("Baseload Deviation", f"{baseload_diff:+.1f}%", delta_color="inverse")
        
        # Create visualization
        fig = go.Figure()
        
        # Add actual usage line
        fig.add_trace(go.Scatter(
            x=day_data_long['Time'],
            y=day_data_long['Usage'],
            name='Actual Usage',
            line=dict(color='#FF8C00', width=2)  # Orange for gas
        ))
        
        # Add baseload reference line
        fig.add_trace(go.Scatter(
            x=day_data_long['Time'],
            y=[day_baseload] * len(day_data_long),
            name='Baseload',
            line=dict(color='#808080', dash='dash')
        ))
        
        # Add anomaly points if any
        anomalies = day_data_long[day_data_long['Anomaly'].notna()]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['Time'],
                y=anomalies['Usage'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10)
            ))
        
        # Get day of week
        day_of_week = pd.to_datetime(selected_date).strftime('%A')
        
        fig.update_layout(
            title=f"Gas Usage vs Baseload - {day_of_week}, {selected_date}",
            xaxis_title="Time",
            yaxis_title="Gas Usage (kWh)",
            hovermode="x unified",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed anomalies
        if not anomalies.empty:
            st.subheader("Baseload Anomaly Report")
            anomaly_df = anomalies.copy()
            anomaly_df['Deviation'] = anomaly_df['Deviation'].round(2)
            st.dataframe(anomaly_df[['Time', 'Usage', 'Baseload', 'Deviation', 'Anomaly']])
            
            # Potential causes and recommendations
            st.subheader("Potential Causes")
            st.markdown("""
            ### Possible reasons for baseload anomalies:
            
            1. **System Controls Issues**
               - Heating controls not properly set back during unoccupied hours
               - Timer or BMS programming errors
               
            2. **Equipment Problems**
               - Continuous running of equipment that should be cycling
               - Valve or damper leakage allowing constant gas flow
               
            3. **Building Envelope**
               - Poor insulation causing excessive heat loss
               - Ventilation systems running unnecessarily
               
            ### Recommended actions:
            
            - Check BMS programming and setbacks
            - Verify boiler operation during low-occupancy periods
            - Inspect equipment for proper cycling behavior
            - Review start-up and shutdown procedures
            """)
    else:
        st.error("No data available for the selected month.")

    st.subheader("📝 Gas Usage Notes")
    
    # Get all dates with data for this month
    available_dates = monthly_data['Date'].dt.date.unique()
    
    if len(available_dates) > 0:
        # Create columns for date selection and notes filtering
        col1, col2 = st.columns([2, 1])
        
        with col1:
            note_date = st.date_input(
                "Select Date for Viewing/Adding Notes",
                value=available_dates[0],
                min_value=min(available_dates),
                max_value=max(available_dates)
            )
        
        with col2:
            # Get dates with notes to show a visual indicator
            dates_with_notes = get_dates_with_notes(selected_hotel, selected_year, selected_month)
            if dates_with_notes:
                st.info(f"{len(dates_with_notes)} dates have notes this month")
        
        # Display existing notes for selected date
        # Display existing notes for selected date
        date_notes = get_notes(selected_hotel, date=note_date)

        if date_notes:
            st.markdown(f"#### Notes for {note_date.strftime('%d %B %Y')}")
            
            # Add filter for resolved notes - Use a unique key for Gas Dashboard
            show_resolved = st.checkbox("Show Resolved Notes", value=True, key="gas_show_resolved_checkbox")
            
            # Filter notes based on checkbox
            display_notes = date_notes if show_resolved else [note for note in date_notes if note.get('status') != 'resolved']
            
            if display_notes:
                for note in display_notes:
                    with st.container():
                        cols = st.columns([5, 1])
                        
                        with cols[0]:
                            # Add visual indicator of resolved status
                            note_bg_color = "#f0f7ff" if note.get('status') != 'resolved' else "#e8f5e9"
                            status_text = " <strong>(Resolved)</strong>" if note.get('status') == 'resolved' else ""
                            
                            st.markdown(f"""
                            <div style="background-color: {note_bg_color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <p>{note['note_text']}</p>
                                <p style="color: #666; font-size: 0.8em;">Added by: {note['created_by']} on {note['created_at']}{status_text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with cols[1]:
                            # Show appropriate action button based on current status
                            if note.get('status') != 'resolved':
                                if st.button("✓ Mark Resolved", key=f"gas_resolve_note_{note['id']}"):
                                    if mark_note_resolved(note['id'], True):
                                        st.success("Note marked as resolved!")
                                        time.sleep(1)
                                        st.rerun()
                            else:
                                if st.button("↻ Reopen", key=f"gas_reopen_note_{note['id']}"):
                                    if mark_note_resolved(note['id'], False):
                                        st.success("Note reopened!")
                                        time.sleep(1)
                                        st.rerun()
            else:
                st.info(f"No {'active ' if not show_resolved else ''}notes available for {note_date.strftime('%d %B %Y')}.")
        else:
            st.info(f"No notes available for {note_date.strftime('%d %B %Y')}.")
        
        # Add a separator
        st.markdown("---")
        
        # Form to add a new note
        st.subheader(f"Add New Note for {note_date.strftime('%d %B %Y')}")
        
        with st.form(key="add_note_form"):
            note_text = st.text_area(
                "Note",
                height=100,
                placeholder="Describe gas usage, reasons for spikes, operational changes, weather conditions, events, etc."
            )
            
            cols = st.columns([3, 1])
            with cols[0]:
                user_name = st.text_input(
                    "Your Name",
                    placeholder="Optional"
                )
            
            with cols[1]:
                submitted = st.form_submit_button("💾 Save Note")
            
            if submitted:
                if not note_text:
                    st.error("Please enter a note before saving.")
                else:
                    created_by = user_name if user_name else "dashboard_user"
                    if add_note(selected_hotel, note_date, note_text, created_by):
                        st.success("Note added successfully!")
                        time.sleep(1)
                        st.rerun()
        
        # Show all notes for the month in an expander
def show_monthly_notes_summary(hotel, year, month, header="📝 Monthly Notes Summary (Add notes in 2nd Tab)"):
    st.subheader(header)
    month_notes = get_notes(hotel, year=year, month=month)

    if month_notes:
        # Add filter for resolved notes in summary view too - with unique key for Gas
        show_resolved_summary = st.checkbox("Show Resolved Notes", value=True, key="gas_show_resolved_summary")
        
        # Filter notes based on checkbox
        display_notes = month_notes if show_resolved_summary else [note for note in month_notes if note.get('status') != 'resolved']
        
        if display_notes:
            with st.expander("View All Notes This Month (add Notes in Real-Time Mgmt Tab)", expanded=False):
                from collections import defaultdict
                notes_by_date = defaultdict(list)
                for note in display_notes:
                    date_str = note['date']
                    notes_by_date[date_str].append(note)

                for date_str, notes in sorted(notes_by_date.items(), reverse=True):
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        st.markdown(f"#### {date_obj.strftime('%d %B %Y')} ({len(notes)} notes)")
                        
                        for note in notes:
                            # Add visual indicator of resolved status
                            note_bg_color = "#f0f7ff" if note.get('status') != 'resolved' else "#e8f5e9"
                            status_text = " <strong>(Resolved)</strong>" if note.get('status') == 'resolved' else ""
                            
                            st.markdown(f"""
                            <div style="background-color: {note_bg_color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <p>{note['note_text']}</p>
                                <p style="color: #666; font-size: 0.8em;">Added by: {note['created_by']} on {note['created_at']}{status_text}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")
                    except ValueError:
                        continue
        else:
            st.info(f"No {'active ' if not show_resolved_summary else ''}notes available for this month.")
    else:
        st.info("No notes available for this month.")

def show_cost_control_tab(monthly_data, hotel_facilities, occupancy_rate):
    """Cost control tab showing financial metrics and analysis"""
    
    st.header("💰 Gas Cost Analysis")

    # Calculate usage and costs
    gas_rate = GAS_PRICE  # £0.05/kWh
    
    total_usage = monthly_data["Total Usage"].sum()
    total_cost = total_usage * gas_rate

    # Key financial metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Total Monthly Cost",
        f"£{total_cost:,.2f}",
        help="Total gas costs"
    )
    
    rooms = hotel_facilities.get('room_count', 100)
    occupied_rooms = rooms * (occupancy_rate/100)
    cost_per_room = total_cost / occupied_rooms if occupied_rooms > 0 else 0
    
    col2.metric(
        "Cost per Occupied Room",
        f"£{cost_per_room:.2f}",
        help=f"Based on {rooms} rooms at {occupancy_rate}% occupancy"
    )
    
    avg_daily_cost = total_cost / len(monthly_data)
    col3.metric(
        "Average Daily Cost",
        f"£{avg_daily_cost:.2f}",
        help="Average cost per day this month"
    )

    # Daily cost pattern
    st.subheader("Daily Gas Cost Pattern")
    
    daily_costs = monthly_data.groupby('Date')['Total Usage'].sum().reset_index()
    daily_costs['Cost'] = daily_costs['Total Usage'] * gas_rate
    
    fig_daily = go.Figure()
    
    fig_daily.add_trace(go.Scatter(
        x=daily_costs['Date'],
        y=daily_costs['Cost'],
        name='Daily Cost',
        mode='lines+markers',
        line=dict(color='#FF8C00')  # Orange for gas
    ))
    
    fig_daily.update_layout(
        title="Daily Gas Cost Trend",
        xaxis_title="Date",
        yaxis_title="Cost (£)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Cost breakdown by time of day
    st.subheader("Gas Usage Pattern by Time of Day")
    
    # Get time columns
    time_cols = [col for col in monthly_data.columns if ':' in col]
    
    # Calculate average usage for each time of day
    time_usage = pd.DataFrame({
        'Time': time_cols,
        'Average Usage': [monthly_data[col].mean() for col in time_cols]
    })
    
    time_usage['Cost'] = time_usage['Average Usage'] * gas_rate
    
    # Create visualization
    fig_time = go.Figure()
    
    fig_time.add_trace(go.Bar(
        x=time_usage['Time'],
        y=time_usage['Average Usage'],
        name='Average Usage',
        marker_color='#FF8C00'  # Orange for gas
    ))
    
    # Add cost line on secondary y-axis
    fig_time.add_trace(go.Scatter(
        x=time_usage['Time'],
        y=time_usage['Cost'],
        name='Average Cost',
        yaxis='y2',
        line=dict(color='#FF4D6D')  # Pink for cost
    ))
    
    fig_time.update_layout(
        title="Average Gas Usage and Cost by Time of Day",
        xaxis_title="Time",
        yaxis_title="Usage (kWh)",
        yaxis2=dict(
            title="Cost (£)",
            overlaying='y',
            side='right'
        ),
        showlegend=True,
        xaxis=dict(
            tickmode='array',
            tickvals=[f"{h:02d}:00" for h in range(0, 24, 3)],
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)]
        )
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

def show_operational_tab(monthly_data, selected_hotel, selected_year, selected_month):
    """Operational efficiency tab showing a heatmap of half-hourly usage patterns"""
    
    st.header("📊 Gas Usage Patterns")

    # Generate a heatmap for half-hourly data
    heatmap_fig = create_heatmap(monthly_data, selected_hotel, selected_year, selected_month)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    ### Understanding the Heatmap
    
    This heatmap shows average gas usage patterns by day of week and time of day.
    
    - **Darker orange areas** indicate higher gas usage
    - **Lighter areas** indicate lower gas usage
    
    #### Key Insights:
    - Look for unexpected usage during unoccupied periods
    - Identify opportunities to reduce gas consumption during off-hours
    - Compare weekday vs weekend patterns
    """)
    
    # Morning/evening usage comparison
    st.subheader("Morning vs. Evening Gas Usage")
    
    # Get time columns
    time_cols = [col for col in monthly_data.columns if ':' in col]
    
    # Define morning and evening hours
    morning_cols = [col for col in time_cols if col.startswith(('05:', '06:', '07:', '08:', '09:'))]
    evening_cols = [col for col in time_cols if col.startswith(('17:', '18:', '19:', '20:', '21:'))]
    
    # Calculate daily averages for morning and evening
    morning_avg = monthly_data[morning_cols].mean().mean()
    evening_avg = monthly_data[evening_cols].mean().mean()
    
    # Display comparison
    col1, col2 = st.columns(2)
    col1.metric("Morning Average (5-10 AM)", f"{morning_avg:.2f} kWh")
    col2.metric("Evening Average (5-10 PM)", f"{evening_avg:.2f} kWh")
    
    # Calculate the ratio
    ratio = evening_avg / morning_avg if morning_avg > 0 else 0
    
    st.markdown(f"**Evening/Morning Ratio**: {ratio:.2f}x")
    
    # Interpretation
    if ratio > 1.5:
        st.info("Evening usage is significantly higher than morning usage, which is typical for properties with evening dining or increased guest occupancy at night.")
    elif ratio < 0.8:
        st.info("Morning usage is higher than evening usage, suggesting focus on breakfast services or morning heating.")
    else:
        st.info("Morning and evening usage are relatively balanced.")

def show_monthly_graph_tab(data, selected_hotel):
    """Monthly Graph tab showing monthly usage across years"""
    st.header("Monthly Gas Usage by Year")
    
    # Filter for selected hotel only
    hotel_data = data[data["Hotel"] == selected_hotel].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Get unique years and sort them
    years = sorted(hotel_data["Year"].unique())
    colors = {years[0]: '#FF8C00', years[-1]: '#FF4D6D'}  # Orange for gas (first year), pink for most recent
    
    for year in years:
        year_data = (hotel_data[hotel_data["Year"] == year]
                    .groupby("Month")["Total Usage"]
                    .sum()
                    .reset_index())
        
        # Sort by month to ensure correct order
        year_data = year_data.sort_values("Month")
        
        # Ensure safe date handling
        if 'Date' in year_data.columns:
            year_data['Date'] = year_data['Date'].apply(
                lambda x: safe_year_replace(x, int(year))
            )
        
        fig.add_trace(go.Bar(
            x=[calendar.month_name[m] for m in year_data["Month"]],
            y=year_data["Total Usage"],
            name=str(year),
            marker_color=colors.get(year, '#808080')
        ))
    
    fig.update_layout(
        title="Monthly Gas Usage by Year",
        xaxis_title="Month",
        yaxis_title="Gas Usage (kWh)",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            title="Year",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary of monthly patterns
    with st.expander("Monthly Gas Usage Patterns"):
        st.markdown("""
        ### Understanding Monthly Gas Usage Patterns
        
        Gas consumption typically follows seasonal patterns:
        
        - **Winter months** (November-February): Higher usage due to heating demands
        - **Summer months** (June-August): Lower usage, primarily for hot water and cooking
        - **Shoulder seasons** (March-May, September-October): Moderate usage with transitional weather
        
        Look for months where your usage pattern deviates from the expected seasonal trend, as this may indicate:
        - Equipment issues or inefficiencies
        - Changes in occupancy patterns
        - Operational changes
        """)
        
        # Calculate month-over-month changes
        if len(years) > 0:
            latest_year = max(years)
            latest_year_data = (hotel_data[hotel_data["Year"] == latest_year]
                               .groupby("Month")["Total Usage"]
                               .sum()
                               .reset_index())
            
            if len(latest_year_data) > 1:
                latest_year_data = latest_year_data.sort_values("Month")
                latest_year_data['Previous Month'] = latest_year_data['Total Usage'].shift(1)
                latest_year_data['MoM Change %'] = ((latest_year_data['Total Usage'] - latest_year_data['Previous Month']) / latest_year_data['Previous Month'] * 100).round(1)
                
                st.subheader(f"Month-over-Month Changes ({latest_year})")
                
                # Format for display
                display_data = latest_year_data.dropna(subset=['MoM Change %']).copy()
                display_data['Month'] = display_data['Month'].apply(lambda x: calendar.month_name[x])
                display_data['MoM Change %'] = display_data['MoM Change %'].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(display_data[['Month', 'Total Usage', 'MoM Change %']])
                
    # Add degree day analysis if available
    st.subheader("Weather Impact Analysis")
    st.info("Feature coming soon: Gas usage vs. Heating Degree Days correlation analysis to normalize for weather impacts.")
    
    # Placeholder for future development
    st.markdown("""
    Weather normalization allows you to:
    - Separate weather impacts from operational efficiency
    - Accurately compare gas usage between time periods with different weather conditions
    - Set realistic targets accounting for weather patterns
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Weather Normalization", "Not Available", "Coming Soon")
    with col2:
        st.metric("Weather-Adjusted Performance", "Not Available", "Coming Soon")

def show_benchmarking_tab(data, monthly_data, selected_hotel, selected_year, selected_month):
    st.header("🎯 Property Benchmarking")

    # Portfolio comparison
    st.subheader("Portfolio Comparison")
    other_hotels = list(set(data['Hotel'].unique()) - {selected_hotel})
    
    benchmark_data = []
    
    # Current hotel data
    current_usage = monthly_data["Total Usage"].sum()
    benchmark_data.append({
        'Hotel': selected_hotel,
        'Total Usage': current_usage
    })
    
    # Add other hotels' data with safe date conversion
    for hotel in other_hotels:
        hotel_mask = (data['Hotel'] == hotel) & (data['Year'] == selected_year) & (data['Month'] == selected_month)
        hotel_data = data[hotel_mask].copy()
        
        # Handle any date conversions safely
        if not hotel_data.empty:
            hotel_data['Date'] = hotel_data['Date'].apply(
                lambda x: safe_year_replace(x, selected_year)
            )
            hotel_usage = hotel_data['Total Usage'].sum()
            benchmark_data.append({
                'Hotel': hotel,
                'Total Usage': hotel_usage
            })

    benchmark_df = pd.DataFrame(benchmark_data)
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=benchmark_df['Hotel'],
        y=benchmark_df['Total Usage'],
        marker_color=[app_colors["highlight"] if hotel == selected_hotel else app_colors["accent"] 
                    for hotel in benchmark_df['Hotel']],
        hovertemplate="<b>%{x}</b><br>" +
                      "Total Gas Usage: %{y:.0f} kWh<extra></extra>"
    ))
    
    fig.update_layout(
        title="Total Gas Usage Comparison",
        xaxis_title="Hotel",
        yaxis_title="Gas Usage (kWh)",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Per room comparison
    st.subheader("Gas Usage Per Room")
    
    # Calculate per room metrics
    for hotel in benchmark_df['Hotel']:
        hotel_facilities = get_hotel_facilities(hotel)
        room_count = hotel_facilities.get('room_count', 100)
        benchmark_df.loc[benchmark_df['Hotel'] == hotel, 'Room Count'] = room_count
    
    benchmark_df['Usage Per Room'] = benchmark_df['Total Usage'] / benchmark_df['Room Count']
    
    # Create visualization
    fig_per_room = go.Figure()
    fig_per_room.add_trace(go.Bar(
        x=benchmark_df['Hotel'],
        y=benchmark_df['Usage Per Room'],
        marker_color=[app_colors["highlight"] if hotel == selected_hotel else app_colors["accent"] 
                    for hotel in benchmark_df['Hotel']],
        hovertemplate="<b>%{x}</b><br>" +
                      "Usage Per Room: %{y:.1f} kWh<br>" +
                      "<extra></extra>"
    ))
    
    fig_per_room.update_layout(
        title="Gas Usage Per Room Comparison",
        xaxis_title="Hotel",
        yaxis_title="Usage Per Room (kWh)",
        showlegend=False
    )
    st.plotly_chart(fig_per_room, use_container_width=True)
    
    # Industry benchmarks
    st.subheader("Industry Benchmarks")
    
    # Define typical gas usage benchmarks by hotel type
    hotel_type = get_hotel_facilities(selected_hotel).get('category', 'Midscale')
    
    benchmarks = {
        "Luxury": {"Low": 12, "Average": 18, "High": 25},
        "Upscale": {"Low": 10, "Average": 15, "High": 20},
        "Midscale": {"Low": 8, "Average": 12, "High": 18},
        "Economy": {"Low": 5, "Average": 8, "High": 12}
    }
    
    # Get the benchmark for the selected hotel type
    hotel_benchmark = benchmarks.get(hotel_type, benchmarks["Midscale"])
    
    # Calculate the hotel's daily average per room
    daily_avg_per_room = 0
    if len(monthly_data) > 0 and benchmark_df[benchmark_df['Hotel'] == selected_hotel]['Room Count'].iloc[0] > 0:
        daily_avg_per_room = (monthly_data['Total Usage'].sum() / len(monthly_data)) / benchmark_df[benchmark_df['Hotel'] == selected_hotel]['Room Count'].iloc[0]
    
    # Display the benchmark comparison
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Your Daily Average",
        f"{daily_avg_per_room:.1f} kWh/room",
        help="Average daily gas usage per room"
    )
    
    col2.metric(
        f"{hotel_type} Average",
        f"{hotel_benchmark['Average']:.1f} kWh/room",
        f"{(daily_avg_per_room - hotel_benchmark['Average']):.1f} kWh",
        help=f"Industry average for {hotel_type} hotels"
    )
    
    # Rating based on performance
    if daily_avg_per_room <= hotel_benchmark['Low']:
        rating = "Excellent"
        description = "Your property is performing better than industry best practices."
    elif daily_avg_per_room <= hotel_benchmark['Average']:
        rating = "Good"
        description = "Your property is performing better than the industry average."
    elif daily_avg_per_room <= hotel_benchmark['High']:
        rating = "Fair"
        description = "Your property is performing within industry norms but has room for improvement."
    else:
        rating = "Needs Improvement"
        description = "Your property is using more gas than industry standards suggest."
    
    col3.metric(
        "Performance Rating",
        rating,
        help=description
    )
    
    # Create benchmark visualization
    st.subheader("Benchmark Visualization")
    
    fig_benchmark = go.Figure()
    
    # Add benchmark ranges
    fig_benchmark.add_trace(go.Bar(
        x=["Low Performer", "Average Performer", "High Performer"],
        y=[hotel_benchmark['High'], hotel_benchmark['Average'], hotel_benchmark['Low']],
        marker_color=['#FF4D6D', '#FFA500', '#00CC66'],
        opacity=0.7,
        name="Industry Benchmarks"
    ))
    
    # Add hotel's performance
    fig_benchmark.add_trace(go.Scatter(
        x=["Your Hotel"],
        y=[daily_avg_per_room],
        mode="markers",
        marker=dict(
            color='#3A3F87',
            size=15,
            line=dict(
                color='white',
                width=2
            )
        ),
        name=selected_hotel
    ))
    
    # Update layout
    fig_benchmark.update_layout(
        title=f"Gas Usage vs. {hotel_type} Hotel Benchmarks",
        xaxis_title="Performance Category",
        yaxis_title="kWh per Room per Day",
        showlegend=True
    )
    
    st.plotly_chart(fig_benchmark, use_container_width=True)

def show_targets_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month):
    st.header("🎯 Monthly Gas Target")
    
    # Get last year's same month as baseline
    baseline_mask = (
        (hotel_data['Hotel'] == selected_hotel) & 
        (hotel_data['Year'] == selected_year - 1) &
        (hotel_data['Month'] == selected_month)
    )
    baseline_usage = hotel_data[baseline_mask]['Total Usage'].sum()
    current_usage = monthly_data['Total Usage'].sum()
    
    # Target setting
    st.subheader("Set Monthly Target")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        target_type = st.selectbox(
            "Target Type",
            ["Percentage Reduction", "Fixed Usage (kWh)"],
            help="Choose how you want to set your target"
        )
    
    with col2:
        if target_type == "Percentage Reduction":
            reduction_target = st.slider(
                f"Reduction Target vs {calendar.month_name[selected_month]} {selected_year-1}",
                min_value=0,
                max_value=30,
                value=10,
                format="%d%%",
                help="Set your reduction target compared to same month last year"
            )
            target_usage = baseline_usage * (1 - reduction_target/100)
        else:
            target_usage = st.number_input(
                "Target Usage (kWh)",
                min_value=0,
                value=int(baseline_usage * 0.9),
                step=1000,
                help="Set your target usage in kWh"
            )
            reduction_target = ((baseline_usage - target_usage) / baseline_usage * 100)
    
    # Progress metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        f"Last Year ({calendar.month_name[selected_month]} {selected_year-1})",
        f"{baseline_usage:,.0f} kWh"
    )
    
    col2.metric(
        f"Target ({calendar.month_name[selected_month]} {selected_year})",
        f"{target_usage:,.0f} kWh",
        f"{-reduction_target:.1f}%",
        help=f"Target reduction from {calendar.month_name[selected_month]} {selected_year-1}"
    )
    
    # Daily tracking
    st.subheader("Daily Progress Tracking")
    
    # Get last year's daily data for comparison with safe date conversion
    last_year_mask = (
        (hotel_data['Hotel'] == selected_hotel) & 
        (hotel_data['Year'] == selected_year - 1) &
        (hotel_data['Month'] == selected_month)
    )
    last_year_daily = hotel_data[last_year_mask].groupby("Date")["Total Usage"].sum().reset_index()
    
    # Use safe date conversion for last year data
    last_year_daily["Date"] = last_year_daily["Date"].apply(
        lambda x: safe_year_replace(x, selected_year)
    )
    
    daily_data = monthly_data.groupby("Date")["Total Usage"].sum().reset_index()
    days_in_month = calendar.monthrange(selected_year, selected_month)[1]
    daily_target = target_usage / days_in_month
    
    fig = go.Figure()
    
    # Current year bars (orange for gas)
    fig.add_trace(go.Bar(
        name=str(selected_year),
        x=daily_data['Date'],
        y=daily_data['Total Usage'],
        marker_color='#FF8C00'  # Orange for gas
    ))
    
    # Last year comparison line
    fig.add_trace(go.Scatter(
        name=str(selected_year-1),
        x=last_year_daily['Date'],
        y=last_year_daily['Total Usage'],
        mode='lines',
        line=dict(color='#FF4D6D')
    ))
    
    # Target line
    fig.add_trace(go.Scatter(
        name='Daily Target',
        x=daily_data['Date'],
        y=[daily_target] * len(daily_data),
        mode='lines',
        line=dict(color='white', dash='dash')
    ))
    
    fig.update_layout(
        title=f"Daily Gas Usage vs Target and Last Year ({calendar.month_name[selected_month]})",
        xaxis_title="Date",
        yaxis_title="Gas Usage (kWh)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Projection warning if month not complete
    if len(daily_data) < days_in_month:
        projected_usage = (current_usage / len(daily_data)) * days_in_month
        projected_diff = ((projected_usage - target_usage) / target_usage) * 100
        st.warning(f"⚠️ Projected end of month usage: {projected_usage:,.0f} kWh ({projected_diff:+.1f}% vs target)")

def show_carbon_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month):
    st.header("🌍 Carbon Emissions Analysis")
    
    # Get appropriate emission factor
    emission_factor = GAS_FACTORS.get(str(selected_year), GAS_FACTORS["2024"])
    
    # Calculate emissions
    monthly_usage = monthly_data['Total Usage'].sum()
    monthly_emissions = monthly_usage * emission_factor
    
    # Calculate YoY comparison with safe date conversion
    prev_year_mask = (
        (hotel_data['Hotel'] == selected_hotel) &
        (hotel_data['Year'] == selected_year - 1) &
        (hotel_data['Month'] == selected_month)
    )
    prev_year_data = hotel_data[prev_year_mask].copy()
    prev_year_usage = prev_year_data['Total Usage'].sum()
    prev_year_factor = GAS_FACTORS.get(str(selected_year - 1), GAS_FACTORS["2023"])
    prev_year_emissions = prev_year_usage * prev_year_factor
    
    col1, col2, col3 = st.columns(3)
    
    # Display metrics
    col1.metric(
        "Monthly Emissions",
        f"{monthly_emissions:,.1f} kgCO2e",
        help=f"Using {emission_factor} kgCO2e/kWh for {selected_year}"
    )
    
    if prev_year_emissions > 0:
        yoy_change = ((monthly_emissions - prev_year_emissions) / prev_year_emissions) * 100
        col2.metric(
            "vs Last Year",
            f"{monthly_emissions:,.1f} kgCO2e",
            f"{yoy_change:+.1f}%",
            delta_color="inverse"
        )
    
    # Emissions per room
    rooms = get_hotel_facilities(selected_hotel).get('room_count', 100)
    emissions_per_room = monthly_emissions / rooms
    col3.metric(
        "Emissions per Room",
        f"{emissions_per_room:,.2f} kgCO2e",
        help=f"Based on {rooms} rooms"
    )
    
    # Emissions trend visualization
    st.subheader("Carbon Emissions Trend")
    
    # Calculate emissions for each month with safe date conversion
    monthly_emissions_data = []
    
    for month in range(1, 13):
        month_mask = (hotel_data['Year'] == selected_year) & (hotel_data['Month'] == month)
        month_data = hotel_data[month_mask].copy()
        usage = month_data['Total Usage'].sum()
        emissions = usage * emission_factor
        
        prev_year_month_mask = (hotel_data['Year'] == selected_year - 1) & (hotel_data['Month'] == month)
        prev_month_data = hotel_data[prev_year_month_mask].copy()
        prev_month_data['Date'] = prev_month_data['Date'].apply(
            lambda x: safe_year_replace(x, selected_year)
        )
        prev_usage = prev_month_data['Total Usage'].sum()
        prev_emissions = prev_usage * prev_year_factor
        
        monthly_emissions_data.append({
            'Month': calendar.month_name[month],
            f'Emissions {selected_year}': emissions,
            f'Emissions {selected_year-1}': prev_emissions
        })
    
    emissions_df = pd.DataFrame(monthly_emissions_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=emissions_df['Month'],
        y=emissions_df[f'Emissions {selected_year}'],
        name=str(selected_year),
        marker_color='#FF8C00'  # Orange for gas
    ))
    
    fig.add_trace(go.Bar(
        x=emissions_df['Month'],
        y=emissions_df[f'Emissions {selected_year-1}'],
        name=str(selected_year-1),
        marker_color='#FF4D6D'
    ))
    
    fig.update_layout(
        title="Monthly Carbon Emissions Comparison",
        xaxis_title="Month",
        yaxis_title="Emissions (kgCO2e)",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Carbon reduction recommendations
    st.subheader("Carbon Reduction Opportunities")
    
    # Define reduction opportunities
    reduction_options = [
        {
            "title": "Smart Controls & Zoning",
            "description": "Install smart thermostats and zone controls to regulate gas usage by area and time of day.",
            "potential_savings": "10-15%",
            "carbon_savings": monthly_emissions * 0.125  # 12.5% average
        },
        {
            "title": "Water Efficiency",
            "description": "Reduce hot water usage through low-flow fixtures and water-efficient practices.",
            "potential_savings": "5-10%",
            "carbon_savings": monthly_emissions * 0.075  # 7.5% average
        },
        {
            "title": "Heating System Optimization",
            "description": "Optimize boiler efficiency through regular maintenance and optimal temperature settings.",
            "potential_savings": "8-12%",
            "carbon_savings": monthly_emissions * 0.1  # 10% average
        }
    ]
    
    # Show options
    for i, option in enumerate(reduction_options):
        with st.expander(f"{option['title']} ({option['potential_savings']} reduction)"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(option['description'])
            with col2:
                st.metric(
                    "Carbon Savings",
                    f"{option['carbon_savings']:,.1f} kgCO2e",
                    help="Estimated monthly carbon reduction"
                )

def show_action_center_tab(monthly_data, hotel_facilities, occupancy_rate):
    st.header("📋 Gas Efficiency Action Center")

    # Quick Wins Section
    st.subheader("💡 UK Gas Efficiency Quick Wins")
    
    # Define potential savings with UK-specific gas measures
    quick_wins = {
        "Boiler Optimization": {
            "actions": [
                "Install weather compensation controls",
                "Optimize flow temperature (max 70°C for condensing boilers)",
                "Implement boiler sequencing for multiple units"
            ],
            "savings": "10-15%",
            "priority": "High",
            "effort": "Low"
        },
        "Building Controls": {
            "actions": [
                "Install TRVs (Thermostatic Radiator Valves) on all radiators",
                "Implement zone control with programmable thermostats",
                "Set heating to follow UK CIBSE guidelines (19-21°C)"
            ],
            "savings": "8-12%",
            "priority": "High",
            "effort": "Medium"
        },
        "Insulation & Fabric": {
            "actions": [
                "Improve pipe insulation to UK Building Regulations standards",
                "Install draught excluders on external doors",
                "Check and upgrade loft insulation (270mm minimum)"
            ],
            "savings": "5-8%",
            "priority": "Medium",
            "effort": "Medium"
        }
    }

    # Add facility-specific actions
    if hotel_facilities.get('has_restaurant'):
        quick_wins["Commercial Kitchen"] = {
            "actions": [
                "Follow CIBSE TM50 guidelines for kitchen ventilation",
                "Install gas-saving devices on commercial hobs",
                "Implement kitchen heat recovery systems"
            ],
            "savings": "12-18%",
            "priority": "High",
            "effort": "Medium"
        }

    # Display actions and calculate potential savings
    col1, col2 = st.columns([3, 2])
    
    with col1:
        selected_actions = {}
        for category, details in quick_wins.items():
            st.markdown(f"**{category}**")
            for action in details['actions']:
                key = f"{category}_{action}"
                selected_actions[key] = st.checkbox(
                    f"{action} ({details['priority']} Priority)",
                    key=key
                )

    with col2:
        st.markdown("### Potential Savings")
        total_savings = 0
        monthly_cost = monthly_data['Total Usage'].sum() * GAS_PRICE  # £0.05/kWh
        
        for category, details in quick_wins.items():
            category_selected = any(
                selected_actions.get(f"{category}_{action}")
                for action in details['actions']
            )
            if category_selected:
                min_savings = float(details['savings'].split('-')[0])
                saving = monthly_cost * (min_savings/100)
                total_savings += saving
                st.write(f"{category}: £{saving:,.2f}")

        st.metric(
            "Total Monthly Savings",
            f"£{total_savings:,.2f}",
            help="Estimated savings based on selected actions"
        )

    # ROI Calculator
    st.subheader("💰 ROI Calculator")
    
    investment = st.number_input(
        "Investment Amount (£)",
        min_value=0.0,
        value=1000.0,
        help="Enter the cost of implementing selected actions"
    )

    if investment > 0:
        payback_period = investment / total_savings if total_savings > 0 else float('inf')
        annual_savings = total_savings * 12
        roi = ((annual_savings - investment) / investment) * 100 if investment > 0 else 0

        roi_col1, roi_col2, roi_col3 = st.columns(3)
        
        roi_col1.metric(
            "Payback Period",
            f"{payback_period:.1f} months",
            help="Time to recover investment"
        )
        
        roi_col2.metric(
            "Annual Savings",
            f"£{annual_savings:,.2f}",
            help="Projected yearly savings"
        )
        
        roi_col3.metric(
            "ROI",
            f"{roi:.1f}%",
            help="First-year return on investment"
        )

    # Implementation Timeline
    st.subheader("📅 Implementation Schedule")
    
    selected_tasks = [
        action for category, details in quick_wins.items()
        for action in details['actions']
        if selected_actions.get(f"{category}_{action}")
    ]

    if selected_tasks:
        fig = go.Figure()
        
        for i, task in enumerate(selected_tasks):
            fig.add_trace(go.Bar(
                x=[2],  # 2 weeks default timeline
                y=[task],
                orientation='h',
                marker_color='#FF8C00'  # Orange for gas
            ))

        fig.update_layout(
            title="Estimated Implementation Timeline",
            xaxis_title="Weeks",
            yaxis_title="Action",
            height=100 + (len(selected_tasks) * 30)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Task Assignment
    st.subheader("👥 Task Assignment")
    
    if selected_tasks:
        for task in selected_tasks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(task)
            with col2:
                st.selectbox(
                    "Assign to",
                    ["Select Team", "Maintenance", "Operations", "Facilities Management"],
                    key=f"assign_{task}"
                )
def convert_fig_to_base64(fig):
    """Convert a Plotly figure to base64 for email embedding"""
    try:
        # For kaleido package issues
        import plotly.io as pio
        
        # Make sure static image rendering is configured
        if not pio.kaleido.scope:
            import kaleido
        
        # Generate the image bytes
        img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
        
        # Encode to base64
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return encoded
    except Exception as e:
        st.sidebar.error(f"Error converting figure to base64: {str(e)}")
        try:
            # Alternative method using io
            import io
            from PIL import Image
            
            # Save figure to a BytesIO object
            buffer = io.BytesIO()
            fig.write_image(buffer, format="png", width=800, height=400, scale=2)
            buffer.seek(0)
            
            # Encode the bytes to base64
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded
        except Exception as e2:
            st.sidebar.error(f"Alternative conversion also failed: {str(e2)}")
            return None

def send_email(recipients, subject, html_content):
    """Send email via SMTP using Gmail with improved recipient handling"""
    try:
        # Gmail SMTP settings - will be replaced by Streamlit secrets in production
        smtp_server = st.secrets["email"]["SMTP_SERVER"]
        smtp_port = int(st.secrets["email"]["SMTP_PORT"])
        sender_email = st.secrets["email"]["SENDER_EMAIL"]
        sender_password = st.secrets["email"]["EMAIL_PASSWORD"]
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"Energy Dashboard <{sender_email}>"
        
        # Process recipients
        if isinstance(recipients, str):
            # Split by newlines and clean up
            recipients = [r.strip() for r in recipients.split('\n') if r.strip()]
        
        # Validate we have at least one recipient
        if not recipients or len(recipients) == 0:
            st.sidebar.error("No valid email recipients provided")
            return False
            
        # Always CC this email
        cc_email = "soraiya.salemohamed@4cgroup.co.uk"
        
        # Set the 'To' field - join all recipients with commas
        msg['To'] = ", ".join(recipients)
        
        # Set the 'Cc' field
        msg['Cc'] = cc_email
        
        # Attach HTML content
        msg.attach(MIMEText(html_content, 'html'))
        
        # Connect to server and send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # Send email to all recipients (including the CC)
        all_recipients = recipients + [cc_email]
        
        # Log what we're sending
        st.sidebar.info(f"Sending email to: {', '.join(all_recipients)}")
        
        # Send the email
        server.sendmail(sender_email, all_recipients, msg.as_string())
        server.quit()
        
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Error sending email: {str(e)}")
        # Print more detailed error info
        import traceback
        st.sidebar.error(traceback.format_exc())
        return False

def generate_email_report(hotel, monthly_data, hotel_data, selected_year, selected_month, 
                         include_usage, include_anomalies, include_heatmap, 
                         include_targets, include_recommendations, anomaly_threshold):
    """Generate the HTML content for the email report"""
    
    # Calculate total usage and cost for executive summary
    total_usage = monthly_data["Total Usage"].sum() if not monthly_data.empty else 0
    rate = GAS_PRICE  # £0.05/kWh for gas
    total_cost = total_usage * rate
    
    # Get YoY data for executive summary
    days_in_current_data = len(monthly_data['Date'].dt.date.unique()) if not monthly_data.empty else 0
    last_year_month_mask = (
        (hotel_data["Year"] == int(selected_year) - 1) & 
        (hotel_data["Month"] == selected_month)
    )
    last_year_month_data = hotel_data[last_year_month_mask]
    
    # Get data for the same number of days
    last_year_usage = 0
    if not last_year_month_data.empty and days_in_current_data > 0:
        last_year_month_data = last_year_month_data.sort_values('Date')
        unique_dates = last_year_month_data['Date'].dt.date.unique()
        if len(unique_dates) >= days_in_current_data:
            last_year_dates = unique_dates[:days_in_current_data]
            last_year_data = last_year_month_data[last_year_month_data['Date'].dt.date.isin(last_year_dates)]
            last_year_usage = last_year_data["Total Usage"].sum()
    
    # YoY comparison for executive summary
    if last_year_usage > 0:
        yoy_change = ((total_usage - last_year_usage) / last_year_usage) * 100
        yoy_text = f"{yoy_change:+.1f}% vs Last Year (same {days_in_current_data} days)"
        yoy_class = "success" if yoy_change < 0 else "alert" if yoy_change > 0 else "info"
    else:
        yoy_text = "No data available for comparison with last year"
        yoy_class = "info"
    
    # Start building the HTML content
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                background-color: #f9f9f9;
            }}
            .header {{
                background-color: #3A3F87;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px 5px 0 0;
            }}
            .section {{
                margin: 20px 0;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metrics-row {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin: 0 -10px;
            }}
            .metric {{
                flex: 1;
                margin: 10px;
                min-width: 180px;
                text-align: center;
                padding: 15px;
                background-color: #f0f0f0;
                border-radius: 5px;
                border-left: 5px solid #3A3F87;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .metric h3 {{
                margin: 0;
                font-size: 14px;
                color: #666;
            }}
            .metric p {{
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
                color: #3A3F87;
            }}
            .metric small {{
                color: #666;
                font-size: 12px;
            }}
            .alert {{
                background-color: #ffebee;
                border-left: 5px solid #f44336;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }}
            .success {{
                background-color: #e8f5e9;
                border-left: 5px solid #4caf50;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }}
            .info {{
                background-color: #e3f2fd;
                border-left: 5px solid #2196f3;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                background-color: white;
            }}
            th, td {{
                padding: 12px 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
                color: #555;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .chart {{
                width: 100%;
                height: 250px;
                background-color: #f5f5f5;
                display: flex;
                justify-content: center;
                align-items: center;
                color: #666;
                border: 1px dashed #ccc;
                margin: 15px 0;
                border-radius: 5px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 12px;
                color: #666;
                padding: 20px;
                background-color: white;
                border-radius: 0 0 5px 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .executive-summary {{
                background-color: white;
                border-left: 5px solid #3A3F87;
                padding: 20px;
                margin: 15px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .highlight {{
                background-color: #fff3cd;
                padding: 2px 5px;
                border-radius: 3px;
                font-weight: bold;
            }}
            .action-items {{
                background-color: #f0f7ff;
                border-radius: 5px;
            }}
            .action-item {{
                margin: 15px 0;
                padding-left: 10px;
                border-left: 3px solid #FF8C00;
            }}
            h2 {{
                color: #3A3F87;
                border-bottom: 2px solid #eaeaea;
                padding-bottom: 8px;
            }}
            .cta-button {{
                background-color: #3A3F87;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
                display: inline-block;
                margin: 15px 0;
                text-align: center;
            }}
            .cta-button:hover {{
                background-color: #2a2e60;
            }}
            .section-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            .section-header h2 {{
                margin: 0;
                border: none;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{hotel} Gas Consumption Report</h1>
            <p>{calendar.month_name[selected_month]} {selected_year}</p>
        </div>
        
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p>This report provides an overview of {hotel}'s gas consumption for {calendar.month_name[selected_month]} {selected_year}. The property used <strong>{total_usage:,.0f} kWh</strong> of gas at an estimated cost of <strong>£{total_cost:,.2f}</strong>.</p>
            <p><span class="highlight">{yoy_text}</span></p>
            <div style="text-align: center;">
                <a href="https://4cgroup-sustainability-dashboard.streamlit.app/Gas_Dashboard" class="cta-button">View Interactive Dashboard</a>
            </div>
        </div>
    """
    
    # Get hotel facilities
    hotel_facilities = get_hotel_facilities(hotel)
    
    # Data validation - check if we have enough data
    days_in_month = calendar.monthrange(int(selected_year), selected_month)[1]
    days_with_data = len(monthly_data['Date'].dt.date.unique())
    data_completeness = (days_with_data / days_in_month) * 100
    
    if data_completeness < 70:  # If less than 70% of days have data
        html_content += f"""
        <div class="alert">
            <strong>⚠️ Incomplete Data Warning:</strong> Data is available for only {days_with_data} of {days_in_month} days in {calendar.month_name[selected_month]}.
            <p>Analysis may not fully represent the month's gas usage patterns.</p>
        </div>
        """
    
    # Usage Summary Section
    if include_usage:
        avg_daily = monthly_data["Total Usage"].mean() if not monthly_data.empty else 0
        
        html_content += f"""
        <div class="section">
            <div class="section-header">
                <h2>Usage Summary</h2>
            </div>
            
            <div class="metrics-row">
                <div class="metric">
                    <h3>Total Usage</h3>
                    <p>{total_usage:,.0f} kWh</p>
                </div>
                
                <div class="metric">
                    <h3>Daily Average</h3>
                    <p>{avg_daily:,.1f} kWh</p>
                </div>
                
                <div class="metric">
                    <h3>Estimated Cost</h3>
                    <p>£{total_cost:,.2f}</p>
                </div>
            </div>
            
            <div class="{yoy_class}">
                <strong>Year-on-Year Comparison:</strong> {yoy_text}
            </div>
        """
        
        # Generate and embed daily usage chart
        try:
            current_daily = monthly_data.groupby("Date")["Total Usage"].sum().reset_index()
            
            last_year_daily = hotel_data[
                (hotel_data["Year"] == int(selected_year) - 1) & 
                (hotel_data["Month"] == selected_month)
            ].groupby("Date")["Total Usage"].sum().reset_index()
            
            # Use safe date conversion
            last_year_daily["Date"] = last_year_daily["Date"].apply(
                lambda x: safe_year_replace(x, int(selected_year))
            )
            
            fig = go.Figure()
            
            # Current year line (orange for gas)
            fig.add_trace(go.Scatter(
                x=current_daily["Date"],
                y=current_daily["Total Usage"],
                mode='lines+markers',
                name=str(selected_year),
                line=dict(color='#FF8C00', width=2),  # Orange for gas
                marker=dict(size=6)
            ))
            
            # Last year line
            fig.add_trace(go.Scatter(
                x=last_year_daily["Date"],
                y=last_year_daily["Total Usage"],
                mode='lines+markers',
                name=str(selected_year - 1),
                line=dict(color='#FF4D6D', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Highlight the highest usage day
            if not current_daily.empty:
                max_idx = current_daily["Total Usage"].argmax()
                fig.add_annotation(
                    x=current_daily["Date"].iloc[max_idx],
                    y=current_daily["Total Usage"].iloc[max_idx],
                    text="Peak usage",
                    showarrow=True,
                    arrowhead=1,
                    bgcolor="#ffffff",
                    bordercolor="#FF4D6D"
                )
            
            # Update layout
            fig.update_layout(
                title=f"Daily Gas Usage: {calendar.month_name[selected_month]} {selected_year} vs {int(selected_year)-1}",
                xaxis_title="Date",
                yaxis_title="Gas Usage (kWh)",
                hovermode="x unified",
                showlegend=True,
                margin=dict(l=30, r=30, t=40, b=30),
                height=300,
                plot_bgcolor='rgba(0,0,0,0.03)'
            )
            
            # Convert to base64
            chart_base64 = convert_fig_to_base64(fig)
            if chart_base64:
                html_content += f"""
                <div>
                    <img src="data:image/png;base64,{chart_base64}" style="width:100%; max-width:800px; border-radius:5px; margin:15px 0;" alt="Daily gas usage chart">
                </div>
                """
            else:
                html_content += """
                <div class="chart">
                    <p>Chart could not be generated</p>
                </div>
                """
        except Exception as e:
            html_content += f"""
            <div class="chart">
                <p>Chart could not be generated: {str(e)}</p>
            </div>
            """
        
        html_content += "</div>"  # Close Usage Summary section
    
    # Operational Heatmap Section
    if include_heatmap:
        html_content += """
        <div class="section">
            <div class="section-header">
                <h2>Operational Efficiency Heatmap</h2>
            </div>
            <p>This heatmap shows your average gas usage by time of day and day of week:</p>
        """
        
        try:
            # This calls your existing dashboard function directly
            heatmap_fig = create_heatmap(
                monthly_data, 
                hotel, 
                int(selected_year), 
                selected_month
            )
            
            # Convert to base64
            heatmap_base64 = convert_fig_to_base64(heatmap_fig)
            
            if heatmap_base64:
                html_content += f"""
                <div>
                    <img src="data:image/png;base64,{heatmap_base64}" style="width:100%; max-width:800px; border-radius:5px; margin:15px 0;" alt="Gas usage heatmap">
                </div>
                """
            else:
                html_content += """
                <div class="chart">
                    <p>Heatmap could not be generated</p>
                </div>
                """
                
            html_content += """
            <div class="info">
                <strong>How to use this chart:</strong>
                <ul>
                    <li>Darker colors indicate higher gas usage</li>
                    <li>Look for unexpected usage patterns during off-hours</li>
                    <li>Compare weekday vs weekend patterns to identify opportunities</li>
                </ul>
            </div>
            """
        except Exception as e:
            html_content += f"""
            <div class="chart">
                <p>Heatmap could not be generated: {str(e)}</p>
            </div>
            """
            
        html_content += """
        <div style="text-align: center;">
            <a href="https://4cgroup-sustainability-dashboard.streamlit.app/Gas_Dashboard" class="cta-button">Explore Full Heatmap Analysis</a>
        </div>
        </div>
        """
    
    # Anomaly Detection Section
    if include_anomalies and days_with_data > 3:  # Only if we have enough data
        # Get time columns
        time_cols = [col for col in monthly_data.columns if ':' in col]
        
        # Simplistic anomaly detection based on daily totals
        daily_totals = monthly_data.groupby(monthly_data['Date'].dt.date)['Total Usage'].sum()
        
        # Calculate mean and standard deviation
        mean_usage = daily_totals.mean()
        std_usage = daily_totals.std()
        
        # Identify anomalies (days with usage more than threshold% away from mean)
        threshold = anomaly_threshold / 100
        anomalies = []
        
        for date, usage in daily_totals.items():
            if abs(usage - mean_usage) / mean_usage > threshold:
                anomaly_type = "High" if usage > mean_usage else "Low"
                deviation = ((usage - mean_usage) / mean_usage) * 100
                anomalies.append({
                    'date': date,
                    'usage': usage,
                    'deviation': deviation,
                    'type': anomaly_type
                })
        
        if anomalies:
            html_content += f"""
            <div class="section">
                <div class="section-header">
                    <h2>Anomaly Detection</h2>
                </div>
                <p>The following significant changes in gas usage were detected (note significant decreases could be caused by missing data from our supplier):</p>
                
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Day</th>
                        <th>Usage</th>
                        <th>vs Average</th>
                        <th>Deviation</th>
                    </tr>
            """
            
            for anomaly in anomalies:
                day_name = pd.Timestamp(anomaly['date']).strftime('%A')
                html_content += f"""
                    <tr>
                        <td>{anomaly['date'].strftime('%d %b')}</td>
                        <td>{day_name}</td>
                        <td>{anomaly['usage']:,.1f} kWh</td>
                        <td>{mean_usage:,.1f} kWh</td>
                        <td style="color: {'red' if anomaly['type'] == 'High' else 'blue'}; font-weight: bold;">{anomaly['deviation']:+.1f}%</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <div class="info">
                    <p><strong>Next Steps:</strong> Investigate these anomalies to understand their causes and take appropriate action.</p>
                </div>
            </div>
            """
        else:
            html_content += f"""
            <div class="section">
                <div class="section-header">
                    <h2>Anomaly Detection</h2>
                </div>
                <div class="success">
                    <p>No significant anomalies detected (threshold: {anomaly_threshold}%).</p>
                </div>
            </div>
            """
    
    # Target Progress Section
    if include_targets:
        # Get last year's same month as baseline
        baseline_mask = (
            (hotel_data['Hotel'] == hotel) & 
            (hotel_data['Year'] == int(selected_year) - 1) &
            (hotel_data['Month'] == selected_month)
        )
        baseline_usage = hotel_data[baseline_mask]['Total Usage'].sum()
        current_usage = monthly_data['Total Usage'].sum()
        
        # Assuming 10% reduction target
        target_reduction = 10
        target_usage = baseline_usage * (1 - target_reduction/100) if baseline_usage > 0 else 0
        
        if target_usage > 0:
            # Calculate progress
            days_in_month = calendar.monthrange(int(selected_year), selected_month)[1]
            days_so_far = days_with_data
            
            projected_usage = (current_usage / days_so_far) * days_in_month if days_so_far > 0 else 0
            
            if projected_usage > 0:
                status_class = "success" if projected_usage <= target_usage else "alert"
                target_diff = abs(projected_usage - target_usage)
                target_status = f"on track to meet target! Projected to be {target_diff:,.0f} kWh under target." if projected_usage <= target_usage else f"at risk of missing target. Projected to exceed by {target_diff:,.0f} kWh."
                
                # Calculate progress percentage
                progress_pct = min(100, (target_usage / projected_usage) * 100) if projected_usage > 0 else 0
                
                html_content += f"""
                <div class="section">
                    <div class="section-header">
                        <h2>Monthly Target Progress</h2>
                    </div>
                    
                    <div class="metrics-row">
                        <div class="metric">
                            <h3>Target Usage</h3>
                            <p>{target_usage:,.0f} kWh</p>
                            <small>{target_reduction}% reduction vs last year</small>
                        </div>
                        
                        <div class="metric">
                            <h3>Current Usage</h3>
                            <p>{current_usage:,.0f} kWh</p>
                            <small>{days_so_far} of {days_in_month} days</small>
                        </div>
                        
                        <div class="metric">
                            <h3>Projected Usage</h3>
                            <p>{projected_usage:,.0f} kWh</p>
                            <small>Based on current average</small>
                        </div>
                    </div>
                    
                    <div class="{status_class}">
                        <strong>Status:</strong> You are {target_status}
                    </div>
                </div>
                """
    
    # Recommendations Section
    if include_recommendations:
        # Get hotel facilities for tailored recommendations
        html_content += """
        <div class="section">
            <div class="section-header">
                <h2>Recommended Actions</h2>
            </div>
            <p>Based on your usage patterns, here are some recommended actions:</p>
            
            <ol style="padding-left: 20px;">
        """
        
        # Standard recommendations for gas
        html_content += """
                <li><strong>Heating Controls</strong>: Implement zone-based heating controls and optimize temperature setpoints (19-21°C for occupied areas).</li>
                <li><strong>Boiler Optimization</strong>: Ensure proper maintenance and optimize flow temperature (max 70°C for condensing boilers).</li>
                <li><strong>Building Fabric</strong>: Check insulation and draught-proofing to reduce heat loss.</li>
        """
        
        # Facility-specific recommendations
        if hotel_facilities.get('has_restaurant', False):
            html_content += """
                <li><strong>Kitchen Operations</strong>: Optimize kitchen equipment scheduling and train staff on efficient cooking practices.</li>
            """
        
        if hotel_facilities.get('has_conf_rooms', False):
            html_content += """
                <li><strong>Conference Facilities</strong>: Implement occupancy-based heating control for meeting rooms.</li>
            """
        
        if hotel_facilities.get('has_pool', False):
            html_content += """
                <li><strong>Pool Management</strong>: Reduce pool water and ambient temperature by 1°C and use pool covers overnight.</li>
            """
        
        html_content += """
            </ol>
            
            <div class="info">
                <p>Implementing these recommendations could save approximately 5-15% on your monthly gas costs.</p>
            </div>
        </div>
        """
    
    # Add Action Items Section with Checkboxes
    html_content += """
    <div class="section action-items">
        <div class="section-header">
            <h2>This Week's Action Items</h2>
        </div>
        <p>Based on this report, we recommend the following actions:</p>
        
        <div class="action-item">
            <input type="checkbox" id="action1">
            <label for="action1"><strong>Review anomalies</strong>: Investigate any unusual usage patterns identified in this report</label>
        </div>
        
        <div class="action-item">
            <input type="checkbox" id="action2">
            <label for="action2"><strong>Share insights</strong>: Discuss findings with your operations team</label>
        </div>
        
        <div class="action-item">
            <input type="checkbox" id="action3">
            <label for="action3"><strong>Implement quick wins</strong>: Select 1-2 energy-saving recommendations to implement this week</label>
        </div>
        
        <div style="text-align: center; margin-top: 20px;">
            <a href="https://4cgroup-sustainability-dashboard.streamlit.app/Gas_Dashboard" class="cta-button">Take Action Now</a>
        </div>
    </div>
    """
    
    # Footer
    dashboard_url = "https://4cgroup-sustainability-dashboard.streamlit.app/Gas_Dashboard"
    html_content += f"""
        <div class="footer">
            <p>This is an automated report from your <a href="{dashboard_url}">Energy Management Dashboard</a>.</p>
            <p>Generated on {datetime.now().strftime('%d %B %Y at %H:%M')}</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def add_email_report_section(selected_hotel, selected_year, selected_month, data):
    """Add an email report section to the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("📧 Email Reports")
    
    # Email the current hotel or select different ones
    report_option = st.sidebar.radio(
        "Generate report for:",
        ["Current hotel only", "Select specific hotels"],
        key="report_option"
    )
    
    if report_option == "Current hotel only":
        hotels_to_report = [selected_hotel]  # Use the currently selected hotel
    else:
        # Let user select multiple hotels
        hotels_to_report = st.sidebar.multiselect(
            "Select hotels for reports:",
            options=sorted(mpan_to_hotel.values()),
            default=[selected_hotel]  # Default to current hotel
        )
    
    # Email recipients
    recipients = st.sidebar.text_area(
        "Email Recipients (one per line)",
        value="",
        placeholder="name@example.com\nmanager@example.com",
        help="Enter email addresses, one per line"
    )
    
    # Report options
    with st.sidebar.expander("Report Content", expanded=False):
        include_usage = st.checkbox("Usage Summary", value=True, key="include_usage")
        include_anomalies = st.checkbox("Anomaly Detection", value=True, key="include_anomalies")
        include_heatmap = st.checkbox("Operational Heatmap", value=True, key="include_heatmap")
        include_targets = st.checkbox("Target Progress", value=True, key="include_targets")
        include_recommendations = st.checkbox("Recommendations", value=True, key="include_recommendations")
        
        anomaly_threshold = st.slider(
            "Anomaly Alert Threshold (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Alert when usage deviates by this percentage"
        )
    
    # Send email button
    if st.sidebar.button("Send Report Now", key="send_report_button"):
        if not hotels_to_report:
            st.sidebar.error("Please select at least one hotel")
            return
        
        if not recipients.strip():
            st.sidebar.error("Please enter at least one email recipient")
            return
        
        # Show progress as reports are generated and sent
        with st.sidebar.status("Generating and sending reports...", expanded=True) as status:
            recipient_list = [r.strip() for r in recipients.strip().split('\n') if r.strip()]
            success_count = 0
            
            for hotel in hotels_to_report:
                status.update(label=f"Processing {hotel}...")
                
                try:
                    # Get hotel data for the report period
                    hotel_mask = (data["Hotel"] == hotel)
                    hotel_all_data = data[hotel_mask].copy()
                    
                    if hotel_all_data.empty:
                        status.update(label=f"No data available for {hotel}", state="error")
                        continue
                    
                    # Get data for the selected year and month
                    report_mask = (
                        (hotel_all_data["Year"] == selected_year) & 
                        (hotel_all_data["Month"] == selected_month)
                    )
                    report_data = hotel_all_data[report_mask].copy()
                    
                    if report_data.empty:
                        status.update(label=f"No data for {hotel} in {calendar.month_name[selected_month]} {selected_year}", state="error")
                        continue
                    
                    # Generate the HTML email content
                    email_html = generate_email_report(
                        hotel,
                        report_data,
                        hotel_all_data,
                        selected_year,
                        selected_month,
                        include_usage,
                        include_anomalies,
                        include_heatmap,
                        include_targets,
                        include_recommendations,
                        anomaly_threshold
                    )
                    
                    # Send the email
                    success = send_email(
                        recipient_list, 
                        f"{hotel} Energy Report - {calendar.month_name[selected_month]} {selected_year}", 
                        email_html
                    )
                    
                    if success:
                        success_count += 1
                    
                except Exception as e:
                    status.update(label=f"Error processing {hotel}: {str(e)}", state="error")
            
            # Update final status
            if success_count == len(hotels_to_report):
                status.update(label=f"Successfully sent reports for all {len(hotels_to_report)} hotels!", state="complete")
            else:
                status.update(label=f"Sent {success_count} of {len(hotels_to_report)} reports", state="error")
def setup_google_sheets():
    """Setup Google Sheets connection"""
    try:
        credentials = {
            "type": "service_account",
            "project_id": st.secrets["gcp_service_account-gas"]["gcp_project_id"],
            "private_key_id": st.secrets["gcp_service_account-gas"]["gcp_private_key_id"],
            "private_key": st.secrets["gcp_service_account-gas"]["gcp_private_key"],
            "client_email": st.secrets["gcp_service_account-gas"]["gcp_client_email"],
            "client_id": st.secrets["gcp_service_account-gas"]["gcp_client_id"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": st.secrets["gcp_service_account-gas"]["gcp_client_x509_cert_url"]
        }
        
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = Credentials.from_service_account_info(credentials, scopes=scopes)
        return gspread.authorize(creds)
        
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        return None

def get_notes_sheet():
    """Get the notes worksheet, creating it if necessary"""
    try:
        client = setup_google_sheets()
        if not client:
            return None
            
        # Try to open the spreadsheet, create if it doesn't exist
        try:
            spreadsheet = client.open("Gas Dashboard Data")
        except gspread.exceptions.SpreadsheetNotFound:
            spreadsheet = client.create("Gas Dashboard Data")
            # Make it accessible to anyone with the link
            spreadsheet.share(None, perm_type='anyone', role='reader')
        
        # Check if notes worksheet exists, create if not
        try:
            worksheet = spreadsheet.worksheet("gas_notes")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title="gas_notes", rows=1000, cols=7)
            # Add headers
            headers = ["id", "hotel", "energy_type", "date", "note_text", "created_by", "created_at"]
            worksheet.update('A1:G1', [headers])
        
        return worksheet
    except Exception as e:
        st.error(f"Error accessing notes sheet: {str(e)}")
        return None

def add_note(hotel, date, note_text, created_by="dashboard_user"):
    """Add a new note for electricity/gas usage"""
    try:
        worksheet = get_notes_sheet()
        if not worksheet:
            return False
        
        # Format date as string if it's a datetime object
        if isinstance(date, (datetime, pd.Timestamp, dt_date)):
            date = date.strftime('%Y-%m-%d')
        
        # Get all records to determine next ID
        records = worksheet.get_all_records()
        next_id = 1
        if records:
            next_id = max([record.get('id', 0) for record in records]) + 1
        
        # Check if status column exists, add it if not
        headers = worksheet.row_values(1)
        if "status" not in headers:
            worksheet.update_cell(1, 8, "status")  # Add status header
        
        # Prepare new record
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_row = [next_id, hotel, 'electricity', date, note_text, created_by, timestamp, "active"]  # Add "active" status
        # For Gas Dashboard, use: new_row = [next_id, hotel, 'gas', date, note_text, created_by, timestamp, "active"]
        
        # Append to the sheet
        worksheet.append_row(new_row)
        
        return True
    except Exception as e:
        st.error(f"Error adding note: {str(e)}")
        return False

def get_notes(hotel, date=None, year=None, month=None, include_resolved=True):
    """Get notes for a specific hotel and date/period"""
    try:
        worksheet = get_notes_sheet()
        if not worksheet:
            return []
        
        # Get all records
        records = worksheet.get_all_records()
        
        # Filter records
        filtered_records = []
        for record in records:
            # For Electricity Dashboard
            if record['hotel'] == hotel and record['energy_type'] == 'gas':  
                # For Gas Dashboard, use: 
                # if record['hotel'] == hotel and record['energy_type'] == 'gas':
                
                # Skip resolved notes if include_resolved is False
                if not include_resolved and record.get('status') == 'resolved':
                    continue
                    
                record_date = record.get('date', '')
                
                # Match specific date
                if date:
                    date_str = date
                    if isinstance(date, (datetime, pd.Timestamp, pd.DatetimeIndex)):
                        date_str = date.strftime('%Y-%m-%d')
                    
                    if record_date == date_str:
                        filtered_records.append(record)
                # Match year and month
                elif year and month:
                    if record_date.startswith(f"{year}-{month:02d}"):
                        filtered_records.append(record)
                # Match year only
                elif year:
                    if record_date.startswith(f"{year}-"):
                        filtered_records.append(record)
                else:
                    filtered_records.append(record)
        
        # Sort by date (descending) and then by created_at (descending)
        filtered_records.sort(key=lambda x: (x['date'], x['created_at']), reverse=True)
        
        return filtered_records
    except Exception as e:
        st.error(f"Error retrieving notes: {str(e)}")
        return []

def delete_note(note_id):
    """Delete a note by ID"""
    try:
        worksheet = get_notes_sheet()
        if not worksheet:
            return False
        
        # Get all records to find the row with matching ID
        records = worksheet.get_all_records()
        
        # Find the row index (adding 2 for header row and 0-indexing)
        row_index = None
        for i, record in enumerate(records):
            if record.get('id') == note_id:
                row_index = i + 2  # +2 for header row and 0-indexing
                break
        
        if row_index:
            worksheet.delete_rows(row_index)
            return True
        
        return False
    except Exception as e:
        st.error(f"Error deleting note: {str(e)}")
        return False
def mark_note_resolved(note_id, resolved=True):
    """Mark a note as resolved or active"""
    try:
        worksheet = get_notes_sheet()
        if not worksheet:
            return False
        
        # Check if status column exists, add it if not
        headers = worksheet.row_values(1)
        if "status" not in headers:
            worksheet.update_cell(1, 8, "status")  # Add status header in column H
        
        # Get all records to find the row with matching ID
        records = worksheet.get_all_records()
        
        # Find the row index (adding 2 for header row and 0-indexing)
        row_index = None
        for i, record in enumerate(records):
            if record.get('id') == note_id:
                row_index = i + 2  # +2 for header row and 0-indexing
                break
        
        if row_index:
            # Update status cell (column 8)
            worksheet.update_cell(row_index, 8, "resolved" if resolved else "active")
            return True
        
        return False
    except Exception as e:
        st.error(f"Error updating note status: {str(e)}")
        return False
def get_dates_with_notes(hotel, year, month):
    """Get all dates in a month that have notes"""
    try:
        # Get notes for the specified month and year
        month_notes = get_notes(hotel, year=year, month=month)
        
        # Extract unique dates
        dates = set()
        for note in month_notes:
            date_str = note.get('date', '')
            if date_str:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                    dates.add(date_obj)
                except ValueError:
                    continue
        
        return sorted(list(dates))
    except Exception as e:
        st.error(f"Error retrieving dates with notes: {str(e)}")
        return []

def main():
    client = setup_google_sheets()
    if not client:
        st.warning("Note-taking functionality may be limited due to Google Sheets configuration issues.")
    st.title("Hotels Gas Consumption Dashboard")
    # Load data and initialize filters
    data = load_data()
    if data is None:
        st.stop()
    
    data = check_for_duplicates(data)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_hotel = st.sidebar.selectbox(
        "Select Hotel",
        options=sorted(data["Hotel"].unique()),
        help="Choose a hotel to view its gas consumption data",
        key="hotel_selectbox"
    )
    
    # Filter data for the selected hotel
    hotel_data = data[data["Hotel"] == selected_hotel].copy()
    
    # Group by Year and Month to see available data periods
    date_check = hotel_data.groupby(['Year', 'Month']).size().reset_index()
  
    latest_date = hotel_data["Date"].max()
    latest_year = latest_date.year
    latest_month = int(latest_date.month)  
    
    years = sorted(hotel_data["Year"].unique())
    # Explicitly convert to list of integers and sort
    months = sorted([int(m) for m in hotel_data["Month"].unique()])
    
    selected_year = st.sidebar.selectbox(
        "Select Year", 
        years,
        index=years.index(latest_year) if latest_year in years else len(years)-1
    )
    
    selected_month = st.sidebar.selectbox(
        "Select Month",
        months,
        index=months.index(latest_month) if latest_month in months else 0,
        format_func=lambda x: calendar.month_name[x],
        key="month_selectbox"
    )
    
    # Filter monthly data based on selected year and month
    monthly_data = hotel_data[(hotel_data["Year"] == selected_year) & (hotel_data["Month"] == selected_month)]

    # Add this section to check for missing dates
    if not monthly_data.empty:
        # Get all dates in the selected month
        start_date = pd.Timestamp(selected_year, selected_month, 1)
        end_date = (start_date + pd.offsets.MonthEnd(1))
        all_dates = pd.date_range(start_date, end_date, freq='D')
        
        # Get actual dates in the data
        actual_dates = monthly_data['Date'].dt.date.unique()
        
        # Find missing dates
        missing_dates = [d for d in all_dates if d.date() not in actual_dates]
        
        # Show warning if there are missing dates
        if missing_dates:
            # Convert to list of dates
            missing_dates = sorted([d.date() for d in missing_dates])
            
            # Group consecutive dates
            date_ranges = []
            range_start = missing_dates[0]
            prev_date = missing_dates[0]
            
            for curr_date in missing_dates[1:] + [None]:
                if curr_date and (curr_date - prev_date).days == 1:
                    prev_date = curr_date
                else:
                    if prev_date == range_start:
                        date_ranges.append(range_start.strftime('%d'))
                    else:
                        date_ranges.append(f"{range_start.strftime('%d')}-{prev_date.strftime('%d')}")
                    if curr_date:
                        range_start = curr_date
                        prev_date = curr_date
            
            missing_dates_str = f"{calendar.month_name[selected_month]} {selected_year}: {', '.join(date_ranges)}"
            st.warning(f"⚠️ Due to Energy Supplier - Missing data for {missing_dates_str}")

    # Get hotel facilities
    hotel_facilities = get_hotel_facilities(selected_hotel)
        
    # Calculate base metrics
    occupancy_rate = st.sidebar.slider(
        "Average Occupancy Rate (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Adjust metrics based on occupancy rate",
        key="occupancy_slider"
    )

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview",
        "🔥 Usage Anomalies",
        "💰 Cost Analysis", 
        "📈 Usage Patterns",
        "📅 Monthly Graph",
        "🎯 Benchmarking",
        "📋 Action Center",
        "🌍 Carbon Emissions"
    ])

    # Display content in each tab
    with tab1:
        show_overview_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month, hotel_facilities, occupancy_rate)
        show_monthly_notes_summary(selected_hotel, selected_year, selected_month)

    with tab2:
        show_realtime_tab(monthly_data, hotel_data, hotel_facilities, occupancy_rate, selected_year, selected_month)
        show_monthly_notes_summary(selected_hotel, selected_year, selected_month)

    with tab3:
        show_cost_control_tab(monthly_data, hotel_facilities, occupancy_rate)

    with tab4:
        show_operational_tab(monthly_data, selected_hotel, selected_year, selected_month)

    with tab5:
        show_monthly_graph_tab(data, selected_hotel)

    with tab6:
        show_benchmarking_tab(data, monthly_data, selected_hotel, selected_year, selected_month)

    with tab7:
        show_action_center_tab(monthly_data, hotel_facilities, occupancy_rate)
    
    with tab8:
        show_carbon_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month)
    
    # Add email report section
    add_email_report_section(selected_hotel, selected_year, selected_month, data)

if __name__ == "__main__":
    main()