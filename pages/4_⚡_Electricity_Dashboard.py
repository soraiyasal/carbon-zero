import streamlit as st
st.set_page_config(
    page_title="Hotels Electricity Dashboard",  # The title shown in browser tab
    page_icon="⚡",       # You can use an emoji or path to an image
    layout="wide",        # 'wide' or 'centered' layout
    initial_sidebar_state="expanded",  # 'expanded' or 'collapsed'
    menu_items={
        'Get Help': 'https://www.streamlit.io/community',
        'Report a bug': "https://github.com/yourusername/yourrepo/issues",
        'About': "# This is a dashboard for monitoring hotel electricity usage."
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
from scipy import stats

from shared_components import add_dashboard_chatbot  # Only this one
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports (industry-standard for energy forecasting)
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ Machine learning libraries not available. Using statistical forecasting only.")


# Add Tawk.to chat widget directly (no import needed)

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


# # Initialize OpenAI client with the API key from secrets
# try:
#     openai.api_key = st.secrets["OPENAI_API_KEY"]
#     st.success("OpenAI API key loaded successfully!")
# except Exception as e:
#     st.error(f"Failed to load OpenAI API key: {str(e)}")
#     st.stop()


# Hide default Streamlit UI elements
# hide_streamlit_style = """
# <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     .stDeployButton {display:none;}
#     .stAlert {display:none;}
#     div[data-testid="stToolbar"] {display: none;}
#     div[data-testid="stDecoration"] {display: none;}
#     div[data-testid="stStatusWidget"] {display: none;}
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define color scheme
app_colors = {
    "background": "#f3f4f6",   # Light gray background
    "header": "#3A3F87",       # Deep blue-violet for headers
    "text": "#2C2E3E",         # Dark charcoal text
    "card_bg": "#ffffff",      # White for cards
    "dropdown_bg": "#f5f7fa",  # Light gray for dropdowns
    "accent": "#3366FF",       # Vibrant blue accent
    "highlight": "#FF4D6D"     # Bold pink-red for highlights
}


# MPAN to Hotel mapping
mpan_to_hotel = {
    "2500021277783": "Westin", 
    "1200051315859": "Camden", 
    "2500021281362": "Canopy",
    "1200052502710": "EH", 
    "1050000997145": "St Albans"
}

# Add this dictionary at the top of your code
ELECTRICITY_FACTORS = {
    "2025":0.18543,
    "2024": 0.20705,
    "2023": 0.207074,
    "2022": 0.19338

}


@st.cache_data(show_spinner=False, ttl=0)  # Change this line to add ttl=0
def load_data():
    try:
        conn = sqlite3.connect('data/electricity_data.db')
        
        # First check what's actually in the database for Camden
        check_query = pd.read_sql_query("""
        SELECT MIN(Date) as min_date, MAX(Date) as max_date 
        FROM hh_data 
        WHERE [Meter Point] = '1200051315859'
        """, conn)
       
        
        # Then load the data with explicit date handling
        data = pd.read_sql_query("""
        SELECT strftime('%Y-%m-%d %H:%M:%S', Date) as Date,
               [Meter Point],
               [Total Usage],
               "00:00","00:30","01:00","01:30","02:00","02:30","03:00","03:30",
               "04:00","04:30","05:00","05:30","06:00","06:30","07:00","07:30",
               "08:00","08:30","09:00","09:30","10:00","10:30","11:00","11:30",
               "12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:30",
               "16:00","16:30","17:00","17:30","18:00","18:30","19:00","19:30",
               "20:00","20:30","21:00","21:30","22:00","22:30","23:00","23:30"
        FROM hh_data
        ORDER BY Date
        """, conn)
        
        # Convert Date column to datetime explicitly
        data['Date'] = pd.to_datetime(data['Date'])
        
        
        # Process like Excel version
        data["Meter Point"] = data["Meter Point"].astype(str)
        data["Hotel"] = data["Meter Point"].map(mpan_to_hotel)
        

        
        # Get time columns
        time_cols = [f"{str(hour).zfill(2)}:{minute}" 
                    for hour in range(24) 
                    for minute in ['00', '30']]
        
        # Ensure numeric values
        for col in time_cols + ['Total Usage']:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
  
        
        # Mirror Excel's processing - use first value, not sum
        grouped_data = data.groupby(['Date', 'Hotel']).agg({
            **{col: 'first' for col in time_cols + ['Total Usage']},
            'Meter Point': 'first'
        }).reset_index()
        
        
        # Add time columns
        grouped_data["Year"] = grouped_data["Date"].dt.year
        grouped_data["Month"] = grouped_data["Date"].dt.month
        grouped_data["Day of Week"] = grouped_data["Date"].dt.day_name()
        
        return grouped_data
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
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
        # Option 1: Keep the first occurrence of each date/hotel combination
        data_deduped = data.drop_duplicates(subset=['Date', 'Hotel'], keep='first')
        
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
    time_cols = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}"
                 for hour in range(24) for i in range(2)]
    
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
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=time_cols,
        y=days_order,
        colorscale="Reds",
        colorbar=dict(title="Usage (kWh)"),
        hoverongaps=False,
        hovertemplate="Day: %{y}<br>Time: %{x}<br>Usage: %{z:.2f} kWh<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Average Half-Hourly Usage by Day - {calendar.month_name[selected_month]} {selected_year}",
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

def get_ai_insights(data_context):
    """Get AI-powered insights with enhanced focus on quick wins"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """
                You are an expert energy analyst specializing in hotel operations and sustainability. 
                Based on industry research and ENERGY STAR guidelines, provide specific, actionable insights focusing on:

                1. Immediate Quick Wins (0-3 months):
                - Room allocation strategies (e.g., booking rooms close together)
                - Temperature and lighting controls
                - Staff and guest behavioral changes
                - Pool and hot water optimization
                
                2. Short-Term Improvements (3-12 months):
                - Equipment maintenance and upgrades
                - Operational schedule optimization
                - Staff training opportunities
                - Guest engagement initiatives
                
                3. Cost-Benefit Analysis:
                - Expected savings per initiative
                - Implementation difficulty (Low/Medium/High)
                - Guest impact assessment
                - Typical payback periods

                4. Performance Tracking:
                - Key metrics to monitor
                - Suggested targets
                - Warning signs to watch for
                - Success indicators

                Use bullet points and be specific about potential savings.
                Include numerical targets where possible.
                Focus on practical, implementable solutions.
                """},
                {"role": "user", "content": f"Analyze this energy usage data and provide insights: {data_context}"}
            ],
            max_tokens=100  # Increased token limit for more detailed response
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"AI analysis currently unavailable: {str(e)}"

def add_enhanced_ai_tab(monthly_data, selected_hotel, selected_year, selected_month):
    """Enhanced AI insights tab with quick wins focus"""
    st.subheader("💡 AI-Powered Energy Insights")
    
    # Calculate base metrics first
    avg_daily = monthly_data["Total Usage"].mean()
    max_daily = monthly_data["Total Usage"].max()
    min_daily = monthly_data["Total Usage"].min()
    
    # Calculate weekday/weekend averages
    weekday_mask = monthly_data["Day of Week"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    weekend_mask = ~weekday_mask
    avg_weekday = monthly_data[weekday_mask]["Total Usage"].mean()
    avg_weekend = monthly_data[weekend_mask]["Total Usage"].mean()
    
    # Calculate total cost
    rate = 0.25
    total_usage = monthly_data["Total Usage"].sum()
    total_cost = total_usage * rate

    # Get hotel context
    hotel_facilities = get_hotel_facilities(selected_hotel)
    
    # Show hotel context
    st.write("### 🏨 Hotel Context")
    context_col1, context_col2 = st.columns(2)
    
    with context_col1:
        st.write("**Hotel Facilities:**")
        st.write(f"• Category: {hotel_facilities.get('category', 'Unknown')}")
        st.write(f"• Room Count: {hotel_facilities.get('room_count', 'Unknown')}")
        st.write(f"• Restaurant: {'Yes' if hotel_facilities.get('has_restaurant') else 'No'}")
    
    with context_col2:
        st.write("**Additional Amenities:**")
        st.write(f"• Swimming Pool: {'Yes' if hotel_facilities.get('has_pool') else 'No'}")
        st.write(f"• Conference Facilities: {'Yes' if hotel_facilities.get('has_conf_rooms') else 'No'}")

    analysis_focus = st.multiselect(
        "Select Areas of Focus:",
        ["Quick Wins", "Operational Changes", "Equipment Optimization", "Guest Experience", "Cost Savings"],
        default=["Quick Wins"],
        key="analysis_focus_select"
    )

    occupancy_rate = st.slider(
        "Average Occupancy Rate (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Adjust to see targeted recommendations based on occupancy",
        key="occupancy_rate_slider"
    )

    if "Quick Wins" in analysis_focus:
        st.subheader("🎯 Quick Wins (0-3 months)")
        
        # Define base quick wins that apply to all hotels
        quick_wins = {
            "Room Management": {
                "actions": [
                    "Book rooms close together by floor/wing",
                    "Implement room vacancy setbacks (18°C winter, 24°C summer)",
                    "Set occupied room temperature to 21°C",
                ],
                "savings": "5-8%",
                "difficulty": "Low",
                "impact_time": "Immediate",
                "requires": None
            },
            "Lighting": {
                "actions": [
                    "Replace bulbs with LEDs as they fail",
                    "Ensure lights are off in unoccupied areas",
                    "Maximize natural daylight use"
                ],
                "savings": "4-6%",
                "difficulty": "Low",
                "impact_time": "Immediate",
                "requires": None
            },
            "Water": {
                "actions": [
                    "Install low-flow shower heads",
                    "Fix any leaking taps",
                    "Optimize hot water temperature (60°C storage, 50°C delivery)"
                ],
                "savings": "3-5%",
                "difficulty": "Low",
                "impact_time": "1 week",
                "requires": None
            },
            "Staff Engagement": {
                "actions": [
                    "Train staff on energy-efficient practices",
                    "Implement a 'switch off' policy",
                    "Regular reporting of energy performance"
                ],
                "savings": "2-4%",
                "difficulty": "Low",
                "impact_time": "1-2 weeks",
                "requires": None
            }
        }

        # Add facility-specific quick wins
        if hotel_facilities.get('has_pool'):
            quick_wins["Pool Operations"] = {
                "actions": [
                    "Reduce pool temperature by 1°C (to 29°C)",
                    "Use pool covers overnight",
                    "Allow pool hall temperature setback overnight to 19°C"
                ],
                "savings": "6-10%",
                "difficulty": "Low",
                "impact_time": "24 hours",
                "requires": "pool"
            }

        if hotel_facilities.get('has_restaurant'):
            quick_wins["Kitchen Operations"] = {
                "actions": [
                    "Optimize kitchen ventilation schedules",
                    "Check and maintain refrigeration seals",
                    "Train staff on equipment usage timing"
                ],
                "savings": "3-5%",
                "difficulty": "Low",
                "impact_time": "1 week",
                "requires": "restaurant"
            }

        if hotel_facilities.get('has_conf_rooms'):
            quick_wins["Conference Facilities"] = {
                "actions": [
                    "Implement occupancy-based HVAC control",
                    "Install motion sensors for lighting",
                    "Optimize pre-heating/cooling schedules"
                ],
                "savings": "2-4%",
                "difficulty": "Low",
                "impact_time": "2 weeks",
                "requires": "conf_rooms"
            }

        # Create savings calculator columns
        st.subheader("💰 Quick Wins Savings Calculator")
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            st.markdown("#### Select Applicable Initiatives")
            selected_initiatives = {}
            for idx, (category, details) in enumerate(quick_wins.items()):
                facility_required = details.get('requires')
                if facility_required is None or hotel_facilities.get(f'has_{facility_required}', False):
                    selected_initiatives[category] = st.checkbox(
                        category, 
                        value=True,
                        key=f"initiative_checkbox_{idx}"
                    )
        with calc_col2:
            st.markdown("#### Estimated Monthly Savings")
            total_savings = 0
            for category, is_selected in selected_initiatives.items():
                if is_selected:
                    min_savings = float(quick_wins[category]["savings"].split("-")[0])
                    saving = total_cost * (min_savings/100)
                    total_savings += saving
                    st.write(f"{category}: £{saving:,.2f}")
            st.markdown(f"#### Total Potential Savings: £{total_savings:,.2f}/month")

        # Add implementation timeline
        st.subheader("📅 Implementation Timeline")
        timeline_data = [(category, 2) for category, is_selected in selected_initiatives.items() if is_selected]
        
        if timeline_data:
            fig = go.Figure()
            
            for initiative, duration in timeline_data:
                fig.add_trace(go.Bar(
                    name=initiative,
                    y=[initiative],
                    x=[duration],
                    orientation='h',
                    marker_color=app_colors["accent"]
                ))

            fig.update_layout(
                title="Weeks to Implement",
                xaxis_title="Weeks",
                showlegend=False,
                height=200 + (len(timeline_data) * 30)
            )

            st.plotly_chart(fig, use_container_width=True)

    # 4. Optional AI Analysis
    if st.toggle("🤖 Enable AI Analysis"):
        with st.expander("🔍 AI-Powered Insights", expanded=True):
            data_context = {
                "average_daily_usage": f"{avg_daily:,.0f} kWh",
                "peak_usage": f"{max_daily:,.0f} kWh",
                "weekday_avg": f"{avg_weekday:,.0f} kWh",
                "weekend_avg": f"{avg_weekend:,.0f} kWh",
                "total_monthly_cost": f"£{total_cost:,.2f}",
                "occupancy_rate": f"{occupancy_rate}%",
                "analysis_focus": analysis_focus,
                "month_year": f"{calendar.month_name[selected_month]} {selected_year}",
                "hotel": selected_hotel,
                "hotel_facilities": hotel_facilities
            }
            
            with st.spinner("🤖 Analyzing your energy data..."):
                ai_insights = get_ai_insights(data_context)
                st.markdown(ai_insights)

            # Add follow-up questions
            user_question = st.text_input(
                "Ask specific questions about implementing these initiatives:",
                placeholder="E.g., How can I optimize room allocation? What's the best way to implement temperature controls?"
            )
            
            if user_question:
                with st.spinner("Generating response..."):
                    follow_up_response = get_ai_insights(
                        f"Question: {user_question}\nContext: {data_context}"
                    )
                    st.markdown(follow_up_response)
def calculate_uk_industry_savings(monthly_data, hotel_facilities, action_type):
    """Calculate savings based on UK industry standards and guidelines"""
    total_usage = monthly_data['Total Usage'].sum()
    
    # UK tariff structure
    peak_rate = 0.25  # £/kWh peak rate
    off_peak_rate = 0.25  # £/kWh off-peak rate
    
    # UK industry standard savings from CIBSE Guide F and Carbon Trust
    uk_savings = {
        "HVAC & Hot Water": {
            "baseline": 0.15,  # 15% baseline based on Carbon Trust
            "actions": [
                "BMS optimization (8-12% savings)",
                "Zone control implementation (5-8% savings)",
                "Heat recovery systems (10-15% savings)"
            ],
            "conditions": {
                "has_pool": 0.05,  # Additional 5% for properties with pools
                "room_count": lambda x: 0.03 if x > 150 else 0
            },
            "investment_range": "£5,000-£15,000",
            "payback": "1-3 years"
        },
        "Lighting Systems": {
            "baseline": 0.12,  # 12% baseline from CIBSE
            "actions": [
                "LED replacement (up to 70% lighting savings)",
                "Occupancy sensors (30-50% area-specific savings)",
                "Daylight linking (20-40% savings in daylit areas)"
            ],
            "conditions": {
                "has_conf_rooms": 0.04,
                "has_restaurant": 0.03
            },
            "investment_range": "£3,000-£10,000",
            "payback": "1-2 years"
        },
        "Kitchen & Catering": {
            "baseline": 0.20,  # 20% baseline for commercial kitchens
            "actions": [
                "Equipment scheduling (10-15% savings)",
                "Extraction control (15-25% savings)",
                "Refrigeration optimization (10-20% savings)"
            ],
            "conditions": {
                "has_restaurant": 0.08  # Additional 8% for full-service
            },
            "investment_range": "£2,000-£8,000",
            "payback": "1-2 years"
        },
        "Building Fabric": {
            "baseline": 0.08,  # 8% baseline from CIBSE
            "actions": [
                "Draught proofing (5-10% savings)",
                "Insulation improvements (10-15% savings)",
                "Glazing optimization (5-10% savings)"
            ],
            "conditions": {
                "room_count": lambda x: 0.04 if x > 200 else 0
            },
            "investment_range": "£10,000-£30,000",
            "payback": "3-5 years"
        }
    }
    
    def calculate_uk_action_savings(action_category):
        savings_pct = uk_savings[action_category]["baseline"]
        
        # Add conditional savings
        for condition, value in uk_savings[action_category]["conditions"].items():
            if condition == "room_count":
                savings_pct += value(hotel_facilities.get("room_count", 0))
            else:
                if hotel_facilities.get(condition, False):
                    savings_pct += value
        
        annual_kwh = total_usage * savings_pct * 12
        annual_cost = (annual_kwh * 0.7 * peak_rate) + (annual_kwh * 0.3 * off_peak_rate)  # Assuming 70/30 peak/off-peak split
        
        return {
            "percentage": savings_pct,
            "annual_kwh": annual_kwh,
            "annual_cost": annual_cost,
            "actions": uk_savings[action_category]["actions"],
            "investment_range": uk_savings[action_category]["investment_range"],
            "payback": uk_savings[action_category]["payback"]
        }
    
    if action_type == "all":
        return {category: calculate_uk_action_savings(category) 
                for category in uk_savings.keys()}
    else:
        return calculate_uk_action_savings(action_type)
def handle_missing_data(data, selected_year, selected_month):
    """Handle missing days in monthly data"""
    # Create complete date range for month
    start_date = pd.Timestamp(selected_year, selected_month, 1)
    end_date = pd.Timestamp(selected_year, selected_month + 1, 1) - pd.Timedelta(days=1)
    complete_dates = pd.DataFrame({
        'Date': pd.date_range(start_date, end_date, freq='D')
    })
    
    # Get last year's dates
    last_year_dates = complete_dates['Date'].map(lambda x: x.replace(year=x.year-1))
    last_year_frame = pd.DataFrame({'Date': last_year_dates})
    
    # Merge current and last year data
    data_reset = data.reset_index() if 'Date' in data.index.names else data.copy()
    current_merged = pd.merge(complete_dates, data_reset, on='Date', how='left')
    last_year_merged = pd.merge(last_year_frame, data_reset, on='Date', how='left')
    
    # Calculate adjustments for missing days
    current_missing = current_merged['Total Usage'].isna().sum()
    last_year_missing = last_year_merged['Total Usage'].isna().sum()
    
    if current_missing > 0 or last_year_missing > 0:
        # Calculate average daily usage for available days
        current_avg = current_merged['Total Usage'].mean()
        last_year_avg = last_year_merged['Total Usage'].mean()
        
        # Fill missing values with averages
        current_merged['Total Usage'] = current_merged['Total Usage'].fillna(current_avg)
        last_year_merged['Total Usage'] = last_year_merged['Total Usage'].fillna(last_year_avg)
        
        # Add missing data indicators
        current_merged['Data Status'] = current_merged['Total Usage'].isna().map({True: 'Estimated', False: 'Actual'})
        last_year_merged['Data Status'] = last_year_merged['Total Usage'].isna().map({True: 'Estimated', False: 'Actual'})
    
    return current_merged, last_year_merged

def show_overview_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month, hotel_facilities, occupancy_rate):
    """Overview tab with current vs last year comparison"""
    
    st.header("📊 Property Overview")
    
    # Calculate metrics based on days in current month
    days_in_month = len(monthly_data)
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
                "Month-to-Date Usage",
                f"{total_usage:,.0f} kWh",
                f"{ytd_change:+.1f}% vs Last Year (same {days_in_month} days)",
                delta_color="inverse",
                help=f"Comparison with same period last year"
            )
        else:
            st.metric(
                "Month-to-Date Usage", 
                f"{total_usage:,.0f} kWh",
                help=f"Usage for first {days_in_month} days"
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
                help="Current average daily consumption"
            )
    
    # Daily comparison visualization
    # Daily comparison visualization
    st.subheader("Daily Usage Comparison")
    
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
    
    # Current year line
    fig.add_trace(go.Scatter(
        x=current_daily["Date"],
        y=current_daily["Total Usage"],
        mode='lines+markers',
        name=str(selected_year),
        line=dict(color='#3366FF', width=2),
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
    
    # ADD THE CODE BLOCK HERE - RIGHT AFTER THE WEEKEND HIGHLIGHTING CODE
    # Add markers for dates with notes
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
        title=f"Daily Usage: {calendar.month_name[selected_month]} {selected_year} vs {selected_year-1} (Weekends Shaded)",
        xaxis_title="Date",
        yaxis_title="Usage (kWh)",
        hovermode="x unified",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    

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

def show_realtime_tab(monthly_data, hotel_data, hotel_facilities, occupancy_rate, selected_year, selected_month):
    """Real-time tab focused on spotting anomalies in energy usage patterns"""
    st.header("⚡ Energy Management - Anomaly Detection")
    
    # Check if we have data
    if monthly_data.empty:
        st.warning("No data available for the selected period.")
        return
        
    # Create tabs for different anomaly detection methods
    anomaly_tab1, anomaly_tab2, anomaly_tab3 = st.tabs(["🔄 Historical Comparison", "⏱️ Baseload Analysis", "📝 Usage Notes"])
    
    # Get selected hotel from monthly data
    selected_hotel = monthly_data["Hotel"].iloc[0] if not monthly_data.empty else None
    
    if selected_hotel is None:
        st.warning("No hotel selected or no data available.")
        return
    
    # Get time columns
    time_cols = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}" 
                for hour in range(24) for i in range(2)]
    # Get time columns
    time_cols = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}" 
                for hour in range(24) for i in range(2)]
    
    with anomaly_tab1:
        # Add baseline explanation
        with st.expander("ℹ️ Understanding Anomaly Detection"):
            st.markdown("""
            ### Anomaly Detection Methods
            
            This dashboard uses two complementary approaches to detect unusual electricity usage:
            
            1. **Historical Comparison**:
               - Compares current patterns with historical averages for similar days
               - Adjusts for day of week (weekday vs weekend) patterns
               - Highlights significant deviations from expected patterns
               
            2. **Baseload Analysis**:
               - Monitors overnight electricity usage (1-5 AM) as your baseload
               - Identifies equipment running unnecessarily during off-hours
               - Monitors weekend/holiday baseload compared to weekday baseload
            
            Anomalies are flagged when:
            - 🔴 **High Anomaly**: Usage > 20% above expected levels
            - 🔵 **Low Anomaly**: Usage > 20% below expected levels
            """)
        
        # Get last year's data with safe conversion
        last_year_mask = (
            (hotel_data["Hotel"] == selected_hotel) & 
            (hotel_data["Year"] == selected_year - 1) & 
            (hotel_data["Month"] == selected_month)
        )
        last_year_data = hotel_data[last_year_mask].copy()
        last_year_data['Date'] = last_year_data['Date'].apply(
            lambda x: safe_year_replace(x, selected_year)
        )
        
        # Date selection with validation
        st.subheader("Select Date for Analysis")
        available_dates = monthly_data['Date'].dt.date.unique()
        
        if len(available_dates) > 0:
            selected_date = st.date_input(
                "Choose a day",
                value=available_dates[0],
                min_value=min(available_dates),
                max_value=max(available_dates),
                key="historical_date_picker"
            )
            
            # Check if data exists for selected date
            selected_day_data = monthly_data[monthly_data['Date'].dt.date == selected_date]
            
            if selected_day_data.empty:
                st.error(f"No data available for {selected_date}. Please select another date.")
                return
            
            # Calculate baseline using statistical method
            baseline_usage = calculate_baseline(monthly_data, last_year_data, time_cols, selected_date)
            
            # Process anomalies using new baseline
            day_data_long = process_anomalies(selected_day_data, time_cols, baseline_usage)
            
            # Display anomaly metrics and visualization
            high_anomalies = day_data_long[day_data_long['Anomaly'] == "High Anomaly"]
            low_anomalies = day_data_long[day_data_long['Anomaly'] == "Low Anomaly"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("High Anomalies", len(high_anomalies))
            with col2:
                st.metric("Low Anomalies", len(low_anomalies))
            
            # Create visualization
            fig = create_anomaly_visualization(day_data_long, selected_date)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed anomalies
            if len(high_anomalies) > 0 or len(low_anomalies) > 0:
                st.subheader("Detailed Anomaly Report")
                anomaly_df = day_data_long[day_data_long['Anomaly'].notna()].copy()
                anomaly_df['Deviation'] = anomaly_df['Deviation'].round(2)
                st.dataframe(anomaly_df[['Time', 'Usage', 'Baseline', 'Deviation', 'Anomaly']])
        else:
            st.error("No data available for the selected month.")
    
    with anomaly_tab2:
        st.subheader("Baseload Analysis")
        
        # Calculate baseload (overnight hours)
        overnight_cols = [col for col in time_cols if col.startswith(('01:', '02:', '03:', '04:', '05:'))]
        
        if not overnight_cols:
            st.error("Unable to calculate baseload - missing overnight data.")
            return
            
        # Calculate baseload for each day
        baseload_data = monthly_data.copy()
        baseload_data['Baseload'] = baseload_data[overnight_cols].mean(axis=1)
        
        # Add day of week and type
        baseload_data['Day of Week'] = baseload_data['Date'].dt.day_name()
        baseload_data['Day Type'] = baseload_data['Date'].dt.dayofweek.apply(
            lambda x: 'Weekend' if x >= 5 else 'Weekday'
        )
        
        # Calculate average baseload by day type
        weekday_baseload = baseload_data[baseload_data['Day Type'] == 'Weekday']['Baseload'].mean()
        weekend_baseload = baseload_data[baseload_data['Day Type'] == 'Weekend']['Baseload'].mean()
        overall_baseload = baseload_data['Baseload'].mean()
        
        # Display average baseload metrics
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Weekday Baseload",
            f"{weekday_baseload:.2f} kWh"
        )
        
        col2.metric(
            "Weekend Baseload",
            f"{weekend_baseload:.2f} kWh",
            f"{((weekend_baseload - weekday_baseload) / weekday_baseload * 100):.1f}% vs Weekday" 
                if weekday_baseload > 0 else None,
            delta_color="inverse"
        )
        
        col3.metric(
            "Overall Baseload",
            f"{overall_baseload:.2f} kWh"
        )
        
        # Baseload efficiency check
        if weekend_baseload > weekday_baseload * 1.1:  # 10% higher
            st.warning("⚠️ Weekend baseload is significantly higher than weekday! Consider reviewing weekend operations.")
        
        # Date selection for baseload
        st.subheader("Daily Baseload Analysis")
        
        if len(available_dates) > 0:
            selected_date = st.date_input(
                "Choose a day",
                value=available_dates[0],
                min_value=min(available_dates),
                max_value=max(available_dates),
                key="baseload_date_picker"
            )
            
            # Filter data for the selected date
            day_data = baseload_data[baseload_data['Date'].dt.date == selected_date]
            
            if not day_data.empty:
                # Get baseload for the selected day
                day_baseload = day_data['Baseload'].iloc[0]
                
                # Get day type average
                day_type = day_data['Day Type'].iloc[0]
                day_type_avg = weekday_baseload if day_type == 'Weekday' else weekend_baseload
                
                # Calculate deviation
                deviation = ((day_baseload - day_type_avg) / day_type_avg * 100) if day_type_avg > 0 else 0
                
                # Display baseload comparison
                col1, col2 = st.columns(2)
                
                col1.metric(
                    f"{day_data['Day of Week'].iloc[0]} Baseload",
                    f"{day_baseload:.2f} kWh"
                )
                
                col2.metric(
                    f"vs Average {day_type}",
                    f"{day_type_avg:.2f} kWh",
                    f"{deviation:+.1f}%",
                    delta_color="inverse"
                )
                
                # Create visualization of day's consumption with baseload reference
                fig = go.Figure()
                
                # Get half-hourly data for the day
                day_data_hourly = day_data.melt(
                    id_vars=['Date'], 
                    value_vars=time_cols, 
                    var_name='Time', 
                    value_name='Usage'
                )
                
                # Sort by time
                day_data_hourly['Hour'] = day_data_hourly['Time'].apply(
                    lambda x: int(x.split(':')[0]) + (0.5 if x.split(':')[1] == '30' else 0)
                )
                day_data_hourly = day_data_hourly.sort_values('Hour')
                
                # Add usage line
                fig.add_trace(go.Scatter(
                    x=day_data_hourly['Time'],
                    y=day_data_hourly['Usage'],
                    name='Actual Usage',
                    line=dict(color='#3366FF', width=2)
                ))
                
                # Add baseload reference
                fig.add_trace(go.Scatter(
                    x=day_data_hourly['Time'],
                    y=[day_baseload] * len(day_data_hourly),
                    name='Baseload',
                    line=dict(color='red', dash='dash')
                ))
                
                # Mark overnight hours
                fig.add_vrect(
                    x0="01:00", x1="05:30",
                    fillcolor="rgba(200, 200, 255, 0.2)",
                    layer="below", line_width=0,
                    annotation_text="Baseload Hours",
                    annotation_position="top left"
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Usage vs Baseload - {day_data['Day of Week'].iloc[0]}, {selected_date}",
                    xaxis_title="Time",
                    yaxis_title="Usage (kWh)",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate baseload percentage
                day_total = day_data[time_cols].sum(axis=1).iloc[0]
                baseload_percentage = (day_baseload * 24) / day_total * 100 if day_total > 0 else 0
                
                # Display baseload insights
                st.subheader("Baseload Insights")
                
                st.write(f"Your baseload represents approximately **{baseload_percentage:.1f}%** of your total daily consumption.")
                
                # Benchmarking against typical values
                hotel_type = get_hotel_facilities(selected_hotel).get('category', 'Midscale')
                typical_values = {
                    "Luxury": {"Low": 15, "Average": 20, "High": 25},
                    "Upscale": {"Low": 12, "Average": 18, "High": 22},
                    "Midscale": {"Low": 10, "Average": 15, "High": 20},
                    "Economy": {"Low": 8, "Average": 12, "High": 18}
                }
                
                typical = typical_values.get(hotel_type, typical_values["Midscale"])
                
                if baseload_percentage < typical["Low"]:
                    st.success(f"✅ Your baseload percentage is excellent - below the typical range for {hotel_type} hotels ({typical['Low']}-{typical['High']}%).")
                elif baseload_percentage < typical["Average"]:
                    st.info(f"ℹ️ Your baseload percentage is good - within the lower half of the typical range for {hotel_type} hotels ({typical['Low']}-{typical['High']}%).")
                elif baseload_percentage < typical["High"]:
                    st.warning(f"⚠️ Your baseload percentage is fair - within the upper half of the typical range for {hotel_type} hotels ({typical['Low']}-{typical['High']}%).")
                else:
                    st.error(f"⚠️ Your baseload percentage is high - above the typical range for {hotel_type} hotels ({typical['Low']}-{typical['High']}%).")
                    
                # Baseload reduction recommendations
                if baseload_percentage > typical["Average"]:
                    st.subheader("Baseload Reduction Recommendations")
                    
                    st.markdown("""
                    ### Recommended Actions:
                    1. **Equipment Audit**: Conduct an overnight audit to identify equipment running unnecessarily
                    2. **Critical vs Non-Critical**: Identify which systems must run 24/7 vs. those that can be scheduled
                    3. **Automation**: Install timers or automated controls for non-critical equipment
                    4. **Maintenance**: Check for equipment operating inefficiently or malfunctioning
                    5. **Staff Training**: Ensure staff follow proper shutdown procedures at the end of shifts
                    """)
            else:
                st.error(f"No data available for {selected_date}. Please select another date.")
        
        # Plot baseload trend
        st.subheader("Monthly Baseload Trend")
        
        daily_baseload = baseload_data.groupby(['Date', 'Day Type'])['Baseload'].mean().reset_index()
        
        fig = go.Figure()
        
        # Add baseload line
        fig.add_trace(go.Scatter(
            x=daily_baseload['Date'],
            y=daily_baseload['Baseload'],
            mode='lines+markers',
            name='Baseload',
            marker=dict(
                color=daily_baseload['Day Type'].map({
                    'Weekday': '#3366FF',
                    'Weekend': '#FF4D6D'
                })
            )
        ))
        
        # Add reference lines
        fig.add_trace(go.Scatter(
            x=[daily_baseload['Date'].min(), daily_baseload['Date'].max()],
            y=[weekday_baseload, weekday_baseload],
            mode='lines',
            name='Avg Weekday Baseload',
            line=dict(color='blue', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=[daily_baseload['Date'].min(), daily_baseload['Date'].max()],
            y=[weekend_baseload, weekend_baseload],
            mode='lines',
            name='Avg Weekend Baseload',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Daily Baseload Trend - {calendar.month_name[selected_month]} {selected_year}",
            xaxis_title="Date",
            yaxis_title="Baseload (kWh)",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        # Notes Tab
    with anomaly_tab3:
        st.subheader("📝 Energy Usage Notes")
        
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
            # Display existing notes for selected date
            date_notes = get_notes(selected_hotel, date=note_date)

            if date_notes:
                st.markdown(f"#### Notes for {note_date.strftime('%d %B %Y')}")
                
                # Use a truly unique key with a UUID-inspired approach
                unique_key = f"daily_notes_{selected_hotel}_{note_date.strftime('%Y%m%d')}"
                show_resolved = st.checkbox("Show Resolved Notes", value=True, key=f"show_resolved_{unique_key}")
                
                # Filter notes based on checkbox
                display_notes = date_notes if show_resolved else [note for note in date_notes if note.get('status') != 'resolved']
                
                if display_notes:
                    for i, note in enumerate(display_notes):
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
                                # Even more unique keys for action buttons
                                button_key = f"note_{note['id']}_{note_date.strftime('%Y%m%d')}_{i}"
                                if note.get('status') != 'resolved':
                                    if st.button("✓ Mark Resolved", key=f"resolve_{button_key}"):
                                        if mark_note_resolved(note['id'], True):
                                            st.success("Note marked as resolved!")
                                            time.sleep(1)
                                            st.rerun()
                                else:
                                    if st.button("↻ Reopen", key=f"reopen_{button_key}"):
                                        if mark_note_resolved(note['id'], False):
                                            st.success("Note reopened!")
                                            time.sleep(1)
                                            st.rerun()
            # Add a separator
            st.markdown("---")
            
            # Form to add a new note
            st.subheader(f"Add New Note for {note_date.strftime('%d %B %Y')}")
            
            with st.form(key="add_note_form"):
                note_text = st.text_area(
                    "Note",
                    height=100,
                    placeholder="Describe energy usage, reasons for spikes, operational changes, weather conditions, events, etc."
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
def show_monthly_notes_summary(hotel, year, month, tab_id="overview", header="📝 Monthly Notes Summary (Add notes in 2nd Tab)"):
    st.subheader(header)
    month_notes = get_notes(hotel, year=year, month=month)

    if month_notes:
        # Add filter for resolved notes in summary view - WITH TAB ID IN THE KEY
        show_resolved_summary = st.checkbox("Show Resolved Notes", value=True, key=f"elec_show_resolved_summary_{tab_id}_{month}_{year}")
        
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
                        
                        for i, note in enumerate(notes):
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

def show_action_center_tab(monthly_data, hotel_facilities, occupancy_rate):
    st.header("📋 Action Center")

    # Quick Wins Section
    st.subheader("💡 Quick Wins")
    
    # Define potential savings
    quick_wins = {
        "Room Management": {
            "actions": [
                "Group bookings by floor/wing",
                "Implement vacancy setbacks (18°C winter, 24°C summer)",
                "Set occupied room temperature to 21°C"
            ],
            "savings": "5-8%",
            "priority": "High",
            "effort": "Low"
        },
        "Lighting Optimization": {
            "actions": [
                "Replace failed bulbs with LEDs",
                "Install motion sensors",
                "Maximize natural light"
            ],
            "savings": "4-6%",
            "priority": "High",
            "effort": "Low"
        },
        "Equipment Maintenance": {
            "actions": [
                "Check HVAC filters",
                "Calibrate thermostats",
                "Inspect door seals"
            ],
            "savings": "3-5%",
            "priority": "Medium",
            "effort": "Medium"
        }
    }

    # Add facility-specific actions
    if hotel_facilities.get('has_restaurant'):
        quick_wins["Kitchen Operations"] = {
            "actions": [
                "Optimize ventilation schedules",
                "Check refrigeration seals",
                "Train staff on equipment timing"
            ],
            "savings": "3-5%",
            "priority": "High",
            "effort": "Low"
        }

    if hotel_facilities.get('has_conf_rooms'):
        quick_wins["Conference Facilities"] = {
            "actions": [
                "Zone-based HVAC control",
                "Occupancy-based lighting",
                "Equipment power management"
            ],
            "savings": "2-4%",
            "priority": "Medium",
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
        monthly_cost = monthly_data['Total Usage'].sum() * 0.25  # Assuming £0.25/kWh

        for category, details in quick_wins.items():
            category_selected = any(
                selected_actions[f"{category}_{action}"]
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
        if selected_actions[f"{category}_{action}"]
    ]

    if selected_tasks:
        fig = go.Figure()
        
        for i, task in enumerate(selected_tasks):
            fig.add_trace(go.Bar(
                x=[2],  # 2 weeks default timeline
                y=[task],
                orientation='h',
                marker_color=app_colors["accent"]
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
                    ["Select Team", "Maintenance", "Operations", "Housekeeping"],
                    key=f"assign_{task}"
                )
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
        line=dict(color='#3366FF', width=2)
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
        title=f"Usage Pattern Analysis - {day_of_week}, {selected_date}",
        xaxis_title="Time",
        yaxis_title="Usage (kWh)",
        hovermode="x unified",
        showlegend=True
    )
    
    return fig

def show_targets_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month):
    st.header("🎯 Monthly Energy Target")
    
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
    
    # Current year bars
    fig.add_trace(go.Bar(
        name=str(selected_year),
        x=daily_data['Date'],
        y=daily_data['Total Usage'],
        marker_color='#3366FF'
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
        title=f"Daily Usage vs Target and Last Year ({calendar.month_name[selected_month]})",
        xaxis_title="Date",
        yaxis_title="Usage (kWh)",
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
    emission_factor = ELECTRICITY_FACTORS.get(str(selected_year), ELECTRICITY_FACTORS["2024"])
    
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
    prev_year_factor = ELECTRICITY_FACTORS.get(str(selected_year - 1), ELECTRICITY_FACTORS["2023"])
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
        marker_color='#3366FF'
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
        yaxis_title="Emissions (tCO2e)",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_cost_control_tab(monthly_data, hotel_facilities, occupancy_rate):
    """Cost control tab showing financial metrics and analysis"""
    
    st.header("💰 Cost Analysis")

    # Calculate peak/off-peak usage and costs
    peak_rate = 0.25  # £/kWh during peak hours
    off_peak_rate = 0.25  # £/kWh during off-peak hours
    
    peak_hours = [f"{str(hour).zfill(2)}:{minute}" 
                for hour in range(7, 22) 
                for minute in ['00', '30']]
    
    off_peak_hours = [f"{str(hour).zfill(2)}:{minute}" 
                    for hour in list(range(0, 7)) + list(range(22, 24))
                    for minute in ['00', '30']]
    
    peak_usage = monthly_data[peak_hours].sum().sum()
    off_peak_usage = monthly_data[off_peak_hours].sum().sum()
    
    peak_cost = peak_usage * peak_rate
    off_peak_cost = off_peak_usage * off_peak_rate
    total_cost = peak_cost + off_peak_cost

    # Key financial metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Total Monthly Cost",
        f"£{total_cost:,.2f}",
        help="Combined peak and off-peak electricity costs"
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

    # Create Cost Analysis tabs
    cost_tab1, cost_tab2, cost_tab3 = st.tabs(["⏰ Peak vs Off-peak", "📈 Daily Trends", "🕒 Time-of-Day"])

    with cost_tab1:
        # Peak vs Off-peak breakdown
        st.subheader("Peak vs Off-peak Analysis")
        
        usage_data = pd.DataFrame({
            'Period': ['Peak (7AM-10PM)', 'Off-Peak (10PM-7AM)'],
            'Usage': [peak_usage, off_peak_usage],
            'Cost': [peak_cost, off_peak_cost],
            'Rate': [peak_rate, off_peak_rate]
        })

        # Create stacked bar visualization
        fig = go.Figure()
        
        # Add usage bars
        fig.add_trace(go.Bar(
            name='Usage',
            x=usage_data['Period'],
            y=usage_data['Usage'],
            yaxis='y',
            marker_color=app_colors["accent"]
        ))
        
        # Add cost line
        fig.add_trace(go.Scatter(
            name='Cost',
            x=usage_data['Period'],
            y=usage_data['Cost'],
            yaxis='y2',
            mode='lines+markers',
            line=dict(color=app_colors["highlight"])
        ))

        fig.update_layout(
            title="Peak vs Off-peak Usage and Cost",
            yaxis=dict(title="Usage (kWh)"),
            yaxis2=dict(title="Cost (£)", overlaying='y', side='right'),
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Calculate percentages for the pie chart
        usage_data['Usage_Percent'] = usage_data['Usage'] / usage_data['Usage'].sum() * 100
        usage_data['Cost_Percent'] = usage_data['Cost'] / usage_data['Cost'].sum() * 100
        
        # Create comparison metrics
        st.subheader("Peak vs Off-peak Distribution")
        
        cols = st.columns(2)
        
        with cols[0]:
            fig_pie_usage = go.Figure(data=[go.Pie(
                labels=usage_data['Period'],
                values=usage_data['Usage'],
                hole=.4,
                marker=dict(colors=['#3366FF', '#808080'])
            )])
            
            fig_pie_usage.update_layout(
                title="Usage Distribution",
                height=350
            )
            
            st.plotly_chart(fig_pie_usage, use_container_width=True)
        
        with cols[1]:
            fig_pie_cost = go.Figure(data=[go.Pie(
                labels=usage_data['Period'],
                values=usage_data['Cost'],
                hole=.4,
                marker=dict(colors=['#3366FF', '#808080'])
            )])
            
            fig_pie_cost.update_layout(
                title="Cost Distribution",
                height=350
            )
            
            st.plotly_chart(fig_pie_cost, use_container_width=True)

        # Cost breakdown table
        st.subheader("Detailed Cost Breakdown")
            
        col1, col2, col3, col4 = st.columns(4)

        breakdown_data = [
            ["Peak Hours", f"{peak_usage:,.2f}", f"£{peak_rate:.2f}", f"£{peak_cost:,.2f}"],
            ["Off-Peak Hours", f"{off_peak_usage:,.2f}", f"£{off_peak_rate:.2f}", f"£{off_peak_cost:,.2f}"],
            ["Total", f"{peak_usage + off_peak_usage:,.2f}", "-", f"£{total_cost:,.2f}"]
        ]

        headers = ["Period", "Usage (kWh)", "Rate (£/kWh)", "Cost (£)"]

        # Create a markdown table
        table_md = "| " + " | ".join(headers) + " |\n"
        table_md += "|" + "|".join(["---"] * len(headers)) + "|\n"
        for row in breakdown_data:
            table_md += "| " + " | ".join(row) + " |\n"

        st.markdown(table_md)
def show_operational_tab(monthly_data, selected_hotel, selected_year, selected_month):
    """Operational efficiency tab showing a heatmap of half-hourly usage patterns"""
    
    st.header("📊 Operational Performance Analysis")

    # Generate a heatmap for half-hourly data
    heatmap_fig = create_heatmap(monthly_data, selected_hotel, selected_year, selected_month)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Add explanation
    with st.expander("ℹ️ Understanding the Heatmap"):
        st.markdown("""
        ### How to Use the Operational Heatmap
        
        This heatmap shows average electricity usage patterns by day of week and time of day.
        
        - **Darker red areas** indicate higher electricity usage
        - **Lighter areas** indicate lower electricity usage
        
        #### Key Insights to Look For:
        - **Off-Hours Usage**: Check for unexpected high usage during nights and weekends
        - **Peak Hours**: Identify when your facility uses the most electricity
        - **Day-to-Day Patterns**: Compare weekday vs weekend patterns
        - **Operational Anomalies**: Look for unusual patterns that don't match expected operations
        
        Use these insights to optimize your building schedules and identify energy-saving opportunities.
        """)
    
    # Morning/evening usage comparison
    st.subheader("Morning vs. Evening Electricity Usage")
    
    # Get time columns
    time_cols = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}" 
                for hour in range(24) for i in range(2)]
    
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
    
    # Create a bar chart for visual comparison
    comparison_data = pd.DataFrame({
        'Period': ['Morning (5-10 AM)', 'Evening (5-10 PM)'],
        'Average Usage': [morning_avg, evening_avg]
    })
    
    fig = go.Figure(data=[
        go.Bar(
            x=comparison_data['Period'],
            y=comparison_data['Average Usage'],
            marker_color=['#3366FF', '#FF4D6D']
        )
    ])
    
    fig.update_layout(
        title="Morning vs. Evening Usage Comparison",
        xaxis_title="Time Period",
        yaxis_title="Average Usage (kWh)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    if ratio > 1.5:
        st.info("🌙 Evening usage is significantly higher than morning usage, which is typical for properties with evening dining or increased guest activities at night. Consider optimizing lighting and HVAC settings during this peak period.")
    elif ratio < 0.8:
        st.info("🌅 Morning usage is higher than evening usage, suggesting focus on breakfast services or morning operations. Review equipment startup sequences and consider staggered startup to reduce morning peak demand.")
    else:
        st.info("⚖️ Morning and evening usage are relatively balanced, indicating consistent operations throughout the day. This is a good pattern that avoids extreme peaks.")
    
    # Hourly pattern throughout the day
    st.subheader("Average Hourly Usage Pattern")
    
    # Group by hour for visualization
    hourly_data = pd.DataFrame({
        'Hour': [hour.split(':')[0] for hour in time_cols if hour.endswith(':00')],
        'Usage': [monthly_data[f"{hour}:00"].mean() for hour in [h.zfill(2) for h in map(str, range(24))]]
    })
    
    # Create line chart
    fig_hourly = go.Figure()
    
    fig_hourly.add_trace(go.Scatter(
        x=hourly_data['Hour'],
        y=hourly_data['Usage'],
        mode='lines+markers',
        line=dict(color='#3366FF', width=2),
        marker=dict(size=8)
    ))
    
    # Add vertical areas for morning and evening
    fig_hourly.add_vrect(
        x0=5, x1=10,
        fillcolor="rgba(100, 149, 237, 0.2)",
        layer="below", line_width=0,
        annotation_text="Morning Period",
        annotation_position="top"  # <-- Changed to "top"
    )
    
    fig_hourly.add_vrect(
        x0=17, x1=22,
        fillcolor="rgba(255, 99, 132, 0.2)",
        layer="below", line_width=0,
        annotation_text="Evening Period",
        annotation_position="top"
    )
    
    fig_hourly.update_layout(
        title="Average Hourly Usage Throughout the Day",
        xaxis_title="Hour of Day",
        yaxis_title="Average Usage (kWh)",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]
        ),
        height=450
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Weekday vs Weekend comparison
    st.subheader("Weekday vs. Weekend Comparison")
    
    # Calculate average daily profile for weekdays and weekends
    weekday_mask = monthly_data["Day of Week"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    weekend_mask = monthly_data["Day of Week"].isin(["Saturday", "Sunday"])
    
    weekday_avg = monthly_data[weekday_mask][time_cols].mean()
    weekend_avg = monthly_data[weekend_mask][time_cols].mean()
    
    # Create DataFrame for plotting
    weekday_weekend_data = pd.DataFrame({
        'Time': time_cols,
        'Weekday': weekday_avg.values,
        'Weekend': weekend_avg.values
    })
    
    # Add hour for sorting
    weekday_weekend_data['Hour'] = weekday_weekend_data['Time'].apply(
        lambda x: int(x.split(':')[0]) + (0.5 if x.split(':')[1] == '30' else 0)
    )
    weekday_weekend_data = weekday_weekend_data.sort_values('Hour')
    
    # Create line chart
    fig_week = go.Figure()
    
    fig_week.add_trace(go.Scatter(
        x=weekday_weekend_data['Time'],
        y=weekday_weekend_data['Weekday'],
        name='Weekday',
        mode='lines',
        line=dict(color='#3366FF', width=2)
    ))
    
    fig_week.add_trace(go.Scatter(
        x=weekday_weekend_data['Time'],
        y=weekday_weekend_data['Weekend'],
        name='Weekend',
        mode='lines',
        line=dict(color='#FF4D6D', width=2)
    ))
    
    fig_week.update_layout(
        title="Weekday vs. Weekend Usage Profile",
        xaxis_title="Time of Day",
        yaxis_title="Average Usage (kWh)",
        xaxis=dict(
            tickmode='array',
            tickvals=[f"{hour:02d}:00" for hour in range(0, 24, 3)],
            ticktext=[f"{hour:02d}:00" for hour in range(0, 24, 3)]
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=450
    )
    
    st.plotly_chart(fig_week, use_container_width=True)
    
    # Calculate percentage difference
    weekday_total = weekday_avg.sum()
    weekend_total = weekend_avg.sum()
    
    if weekday_total > 0:
        weekend_diff = ((weekend_total - weekday_total) / weekday_total) * 100
        
        if weekend_diff < -20:
            st.success(f"✅ Weekend usage is {abs(weekend_diff):.1f}% lower than weekday usage, showing good operational efficiency during non-business days.")
        elif weekend_diff > 10:
            st.warning(f"⚠️ Weekend usage is {weekend_diff:.1f}% higher than weekday usage, which is unusual. Consider reviewing weekend operations.")
        else:
            st.info(f"ℹ️ Weekend usage is {weekend_diff:+.1f}% compared to weekday usage, which is within normal range.")
    else:
        st.info("ℹ️ Unable to calculate weekday-weekend comparison due to insufficient data.")

def show_monthly_graph_tab(data, selected_hotel):
    """Monthly Graph tab showing monthly usage across years with weather normalization"""
    st.header("Monthly Usage Analysis")
    
    # Filter for selected hotel only
    hotel_data = data[data["Hotel"] == selected_hotel].copy()
    
    # Create tabs for different views
    graph_tab1, graph_tab2 = st.tabs(["📊 Monthly Comparison", "🌡️ Weather Impact"])
    
    with graph_tab1:
        # Create figure
        fig = go.Figure()
        
        # Get unique years and sort them
        years = sorted(hotel_data["Year"].unique())
        colors = {years[0]: '#3366FF', years[-1]: '#FF4D6D'}
        
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
            title="Monthly Usage by Year",
            xaxis_title="Month",
            yaxis_title="Usage (kWh)",
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
        
        if len(years) > 1:
            # Year-over-year changes
            st.subheader("Year-over-Year Changes")
            
            latest_year = max(years)
            prev_year = latest_year - 1
            
            if prev_year in years:
                # Calculate YoY changes for each month
                latest_year_data = hotel_data[hotel_data["Year"] == latest_year]
                prev_year_data = hotel_data[hotel_data["Year"] == prev_year]
                
                # Group by month for both years
                latest_monthly = latest_year_data.groupby("Month")["Total Usage"].sum()
                prev_monthly = prev_year_data.groupby("Month")["Total Usage"].sum()
                
                # Get common months to avoid shape mismatches
                common_months = sorted(set(latest_monthly.index).intersection(set(prev_monthly.index)))
                
                if common_months:
                    # Build DataFrame with only matching months
                    yoy_changes = []
                    
                    for month in common_months:
                        if month in latest_monthly.index and month in prev_monthly.index:
                            current_value = latest_monthly[month]
                            prev_value = prev_monthly[month]
                            
                            if prev_value > 0:
                                change_pct = ((current_value - prev_value) / prev_value * 100)
                            else:
                                change_pct = float('nan')
                                
                            yoy_changes.append({
                                'Month': calendar.month_name[month],
                                'Current': current_value,
                                'Previous': prev_value,
                                'Change %': round(change_pct, 1)
                            })
                    
                    if yoy_changes:
                        changes_df = pd.DataFrame(yoy_changes)
                        
                        # Style DataFrame for display
                        def highlight_changes(val):
                            if isinstance(val, float):
                                if val < -5:
                                    return 'background-color: #a8f0c6; color: black'  # Green for good
                                elif val > 5:
                                    return 'background-color: #f7a9a9; color: black'  # Red for bad
                            return ''
                        
                        styled_changes = changes_df.style.format({
                            'Current': '{:,.0f}'.format,
                            'Previous': '{:,.0f}'.format,
                            'Change %': '{:+.1f}%'.format
                        }).applymap(highlight_changes, subset=['Change %'])
                        
                        st.dataframe(styled_changes, use_container_width=True)
                    else:
                        st.info("No common months found for year-over-year comparison.")
                else:
                    st.info("No common months found for year-over-year comparison.")
    
    with graph_tab2:
        # st.subheader("Weather Normalization Analysis")
        
        # # Simulated HDD data (would be replaced with actual heating degree days)
        # months = list(range(1, 13))
        # hdd_data = pd.DataFrame({
        #     'Month': months,
        #     'HDD': [456, 390, 320, 250, 170, 100, 80, 90, 140, 220, 340, 410],
        #     'Month_Name': [calendar.month_name[m] for m in months]
        # })
        
        # # Create scatter plot for electricity vs HDD
        # st.write("#### Electricity Usage vs Heating Degree Days")
        
        # # Get monthly usage data for the latest year
        # latest_year = max(hotel_data["Year"].unique())
        # monthly_usage = hotel_data[hotel_data["Year"] == latest_year].groupby("Month")["Total Usage"].sum().reset_index()
        
        # # Merge with HDD data
        # merged_data = pd.merge(monthly_usage, hdd_data, on="Month")
        
        # # Create scatter plot
        # fig_scatter = px.scatter(
        #     merged_data,
        #     x="HDD",
        #     y="Total Usage",
        #     text="Month_Name",
        #     trendline="ols",
        #     labels={"HDD": "Heating Degree Days", "Total Usage": "Electricity Usage (kWh)"}
        # )
        
        # fig_scatter.update_traces(
        #     textposition='top center',
        #     marker=dict(size=12, color='#3366FF')
        # )
        
        # fig_scatter.update_layout(
        #     title="Electricity Usage vs Weather (HDDs)",
        #     xaxis_title="Heating Degree Days",
        #     yaxis_title="Electricity Usage (kWh)"
        # )
        
        # st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.info("Feature coming soon")
        
        # # Weather normalization explanation
        # st.markdown("""
        # ### Weather Normalization Benefits
        
        # Weather normalization allows you to:
        # - Separate weather-dependent consumption from operational inefficiencies
        # - Make accurate year-to-year comparisons regardless of weather differences
        # - Set realistic targets accounting for temperature variations
        # - Identify when high usage is due to unexpected weather vs operational issues
        
        # The correlation between usage and HDDs helps quantify your building's weather sensitivity.
        # """)
        
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.metric("Current Weather Normalization", "Not Available", "Coming Soon")
        # with col2:
        #     st.metric("Weather-Adjusted Performance", "Not Available", "Coming Soon")
def show_benchmarking_tab(data, monthly_data, selected_hotel, selected_year, selected_month):
    st.header("🎯 Performance Benchmarking")

    # Create benchmarking tabs
    bench_tab1, bench_tab2, bench_tab3 = st.tabs(["🏨 Portfolio Comparison", "🔍 Industry Standards", "✅ UK Compliance"])
    
    with bench_tab1:
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
                        "Total Usage: %{y:.0f} kWh<extra></extra>"
        ))
        
        fig.update_layout(
            title="Total Usage Comparison",
            xaxis_title="Hotel",
            yaxis_title="Usage (kWh)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Per room comparison
        st.subheader("Usage Per Room")
        
        # Calculate per room metrics
        benchmark_df['Room Count'] = benchmark_df['Hotel'].apply(
            lambda x: get_hotel_facilities(x).get('room_count', 100)
        )
        
        benchmark_df['Usage Per Room'] = benchmark_df['Total Usage'] / benchmark_df['Room Count']
        
        fig_per_room = go.Figure()
        
        fig_per_room.add_trace(go.Bar(
            x=benchmark_df['Hotel'],
            y=benchmark_df['Usage Per Room'],
            marker_color=[app_colors["highlight"] if hotel == selected_hotel else app_colors["accent"] 
                        for hotel in benchmark_df['Hotel']],
            hovertemplate="<b>%{x}</b><br>" +
                        "Usage Per Room: %{y:.1f} kWh<extra></extra>"
        ))
        
        fig_per_room.update_layout(
            title="Usage Per Room Comparison",
            xaxis_title="Hotel",
            yaxis_title="Usage Per Room (kWh)",
            showlegend=False
        )
        
        st.plotly_chart(fig_per_room, use_container_width=True)
        
        # Ranking
        st.subheader("Portfolio Ranking")
        
        # Rank hotels by efficiency (usage per room)
        ranked_df = benchmark_df.sort_values('Usage Per Room')
        ranked_df['Rank'] = range(1, len(ranked_df) + 1)
        
        # Find rank of selected hotel
        selected_rank = ranked_df[ranked_df['Hotel'] == selected_hotel]['Rank'].iloc[0]
        total_hotels = len(ranked_df)
        
        # Create ranking metrics
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Current Rank", 
            f"{selected_rank} of {total_hotels}",
            help="Lower rank is better (less energy per room)"
        )
        
        col2.metric(
            "Best Performer",
            ranked_df.iloc[0]['Hotel'],
            f"{ranked_df.iloc[0]['Usage Per Room']:.1f} kWh/room",
            help="Hotel with lowest energy use per room"
        )
        
        average_usage = benchmark_df['Usage Per Room'].mean()
        diff_from_avg = ((benchmark_df[benchmark_df['Hotel'] == selected_hotel]['Usage Per Room'].iloc[0] - average_usage) / average_usage) * 100
        
        col3.metric(
            "vs. Portfolio Average",
            f"{diff_from_avg:+.1f}%",
            delta_color="inverse",
            help="Percentage difference from portfolio average"
        )
        
    with bench_tab2:
        st.subheader("Industry Standards Comparison")
        
        # Define typical electricity usage benchmarks by hotel type
        hotel_type = get_hotel_facilities(selected_hotel).get('category', 'Midscale')
        
        benchmarks = {
            "Luxury": {"Low": 15, "Average": 25, "High": 35},
            "Upscale": {"Low": 12, "Average": 20, "High": 28},
            "Midscale": {"Low": 8, "Average": 15, "High": 22},
            "Economy": {"Low": 5, "Average": 10, "High": 15}
        }
        
        # Get the benchmark for the selected hotel type
        hotel_benchmark = benchmarks.get(hotel_type, benchmarks["Midscale"])
        
        # Calculate the hotel's daily average per room
        daily_avg_per_room = 0
        if len(monthly_data) > 0 and 'Room Count' in benchmark_df.columns:
            room_count = benchmark_df[benchmark_df['Hotel'] == selected_hotel]['Room Count'].iloc[0]
            if room_count > 0:
                daily_avg_per_room = (monthly_data['Total Usage'].sum() / len(monthly_data)) / room_count
        
        # Display the benchmark comparison
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Your Daily Average",
            f"{daily_avg_per_room:.1f} kWh/room",
            help="Average daily electricity usage per room"
        )
        
        col2.metric(
            f"{hotel_type} Average",
            f"{hotel_benchmark['Average']:.1f} kWh/room",
            f"{(daily_avg_per_room - hotel_benchmark['Average']):.1f} kWh",
            delta_color="inverse",
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
            description = "Your property is using more electricity than industry standards suggest."
        
        col3.metric(
            "Performance Rating",
            rating,
            help=description
        )
        
        # Create benchmark visualization
        st.subheader("Benchmark Comparison")
        
        fig_benchmark = go.Figure()
        
        # Add benchmark ranges as a bar chart with colored segments
        benchmark_data = pd.DataFrame([
            {'Category': 'Industry Low', 'Value': hotel_benchmark['Low'], 'Color': '#00CC66'},
            {'Category': 'Industry Average', 'Value': hotel_benchmark['Average'] - hotel_benchmark['Low'], 'Color': '#FFA500'},
            {'Category': 'Industry High', 'Value': hotel_benchmark['High'] - hotel_benchmark['Average'], 'Color': '#FF4D6D'}
        ])
        
        # Create a stacked bar for the benchmark categories
        fig_benchmark.add_trace(go.Bar(
            y=['Industry Benchmark'],
            x=[hotel_benchmark['Low']],
            name='Excellent',
            orientation='h',
            marker=dict(color='#00CC66'),
            width=0.5
        ))
        
        fig_benchmark.add_trace(go.Bar(
            y=['Industry Benchmark'],
            x=[hotel_benchmark['Average'] - hotel_benchmark['Low']],
            name='Good',
            orientation='h',
            marker=dict(color='#FFA500'),
            width=0.5,
            base=[hotel_benchmark['Low']]
        ))
        
        fig_benchmark.add_trace(go.Bar(
            y=['Industry Benchmark'],
            x=[hotel_benchmark['High'] - hotel_benchmark['Average']],
            name='Fair',
            orientation='h',
            marker=dict(color='#FF4D6D'),
            width=0.5,
            base=[hotel_benchmark['Average']]
        ))
        
        # Add your hotel as a marker
        fig_benchmark.add_trace(go.Scatter(
            y=['Your Hotel'],
            x=[daily_avg_per_room],
            mode='markers',
            marker=dict(
                color='#3366FF',
                size=15,
                symbol='diamond'
            ),
            name=selected_hotel
        ))
        
        # Add vertical lines for benchmarks
        fig_benchmark.add_vline(
            x=hotel_benchmark['Low'],
            line_width=1, line_dash="dash", line_color="#00CC66"
        )
        
        fig_benchmark.add_vline(
            x=hotel_benchmark['Average'],
            line_width=1, line_dash="dash", line_color="#FFA500"
        )
        
        fig_benchmark.add_vline(
            x=hotel_benchmark['High'],
            line_width=1, line_dash="dash", line_color="#FF4D6D"
        )
        
        # Add annotations
        fig_benchmark.add_annotation(
            x=hotel_benchmark['Low'] / 2,
            y='Industry Benchmark',
            text="Excellent",
            showarrow=False,
            font=dict(color='white')
        )
        
        fig_benchmark.add_annotation(
            x=hotel_benchmark['Low'] + (hotel_benchmark['Average'] - hotel_benchmark['Low']) / 2,
            y='Industry Benchmark',
            text="Good",
            showarrow=False,
            font=dict(color='white')
        )
        
        fig_benchmark.add_annotation(
            x=hotel_benchmark['Average'] + (hotel_benchmark['High'] - hotel_benchmark['Average']) / 2,
            y='Industry Benchmark',
            text="Fair",
            showarrow=False,
            font=dict(color='white')
        )
        
        # Update layout
        fig_benchmark.update_layout(
            title=f"Electricity Usage vs. {hotel_type} Hotel Benchmarks",
            xaxis_title="kWh per Room per Day",
            showlegend=False,
            barmode='stack',
            height=300,
            margin=dict(t=50, b=50, l=20, r=20)
        )
        
        st.plotly_chart(fig_benchmark, use_container_width=True)
        
        st.markdown(f"""
        ### Benchmark Interpretation
        
        Your property is rated as **{rating}** based on industry benchmarks for {hotel_type} hotels:
        
        - **Excellent**: Less than {hotel_benchmark['Low']} kWh/room/day
        - **Good**: {hotel_benchmark['Low']} - {hotel_benchmark['Average']} kWh/room/day
        - **Fair**: {hotel_benchmark['Average']} - {hotel_benchmark['High']} kWh/room/day
        - **Needs Improvement**: Greater than {hotel_benchmark['High']} kWh/room/day
        
        {description}
        """)
    
    with bench_tab3:
        st.subheader("UK Compliance & Reporting")
        
        # ESOS compliance information
        st.markdown("""
        ### ESOS Compliance Status
        
        **Energy Savings Opportunity Scheme (ESOS)** is mandatory for large UK organizations. The current phase (Phase 3) compliance deadline was 5 December 2023.
        """)
        
        # Create a visual indicator of compliance status
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### Status:")
        
        with col2:
            st.success("✅ COMPLIANT - Next assessment due by December 2027")
        
        # SECR information
        st.markdown("""
        ### Streamlined Energy and Carbon Reporting (SECR)
        
        Large UK companies are required to disclose energy usage, greenhouse gas emissions, and energy efficiency actions in their annual reports.
        """)
        
        # Placeholder for SECR numbers
        secr_col1, secr_col2, secr_col3 = st.columns(3)
        
        # Calculate annual consumption based on current month
        annual_estimate = monthly_data['Total Usage'].sum() * 12  # Simple multiplication for now
        
        secr_col1.metric(
            "Annual Consumption", 
            f"{annual_estimate:,.0f} kWh",
            help="Estimated annual consumption based on current usage"
        )
        
        # Calculate emissions using UK factors
        annual_emissions = annual_estimate * ELECTRICITY_FACTORS.get(str(selected_year), 0.2)
        
        secr_col2.metric(
            "Annual Emissions",
            f"{annual_emissions/1000:,.1f} tCO2e",
            help=f"Based on {selected_year} UK electricity carbon factor"
        )
        
        secr_col3.metric(
            "Emissions Intensity",
            f"{annual_emissions/get_hotel_facilities(selected_hotel).get('room_count', 100)/365:,.2f} kgCO2e/room/day",
            help="Emissions per room per day - key SECR metric"
        )
        
        # Energy efficiency actions
        st.markdown("""
        ### SECR Reportable Energy Efficiency Actions
        
        The following energy efficiency actions should be included in your annual SECR report:
        """)
        
        actions_col1, actions_col2 = st.columns(2)
        
        with actions_col1:
            st.markdown("""
            #### Implemented Actions
            - LED lighting upgrades
            - BMS optimization and controls
            - Staff energy awareness training
            - Energy management policy implementation
            """)
        
        with actions_col2:
            st.markdown("""
            #### Planned Actions
            - Solar PV feasibility study
            - HVAC equipment upgrades
            - ISO 50001 energy management certification
            - Overnight shutdown procedure implementation
            """)
        
        # Recommendations
        st.markdown("""
        ### UK Compliance Recommendations
        
        1. **Maintain Records**: Keep detailed energy consumption records for ESOS Phase 4
        2. **Track Intensity Metrics**: Monitor kWh/room/night for benchmarking and reporting
        3. **Document Actions**: Record all energy efficiency initiatives for SECR reporting
        4. **Consider Display Energy Certificates (DECs)**: Voluntary for hotels but good practice
        5. **Monitor Regulatory Changes**: Stay updated on UK energy compliance requirements
        """)
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
    rate = 0.25  # £/kWh
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
                border-left: 3px solid #3366FF;
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
            <h1>{hotel} Energy Report</h1>
            <p>{calendar.month_name[selected_month]} {selected_year}</p>
        </div>
        
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p>This report provides an overview of {hotel}'s energy performance for {calendar.month_name[selected_month]} {selected_year}. The property used <strong>{total_usage:,.0f} kWh</strong> of electricity at an estimated cost of <strong>£{total_cost:,.2f}</strong>.</p>
            <p><span class="highlight">{yoy_text}</span></p>
            <div style="text-align: center;">
                <a href="https://4cgroup-sustainability-dashboard.streamlit.app/Electricity_Dashboard" class="cta-button">View Interactive Dashboard</a>
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
            <p>Analysis may not fully represent the month's energy usage patterns.</p>
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
            
            # Current year line
            fig.add_trace(go.Scatter(
                x=current_daily["Date"],
                y=current_daily["Total Usage"],
                mode='lines+markers',
                name=str(selected_year),
                line=dict(color='#3366FF', width=2),
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
                title=f"Daily Usage: {calendar.month_name[selected_month]} {selected_year} vs {int(selected_year)-1}",
                xaxis_title="Date",
                yaxis_title="Usage (kWh)",
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
                    <img src="data:image/png;base64,{chart_base64}" style="width:100%; max-width:800px; border-radius:5px; margin:15px 0;" alt="Daily usage chart">
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
            <p>This heatmap shows your average energy usage by time of day and day of week:</p>
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
                    <img src="data:image/png;base64,{heatmap_base64}" style="width:100%; max-width:800px; border-radius:5px; margin:15px 0;" alt="Energy usage heatmap">
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
                    <li>Darker colors indicate higher energy usage</li>
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
            <a href="https://4cgroup-sustainability-dashboard.streamlit.app/Electricity_Dashboard" class="cta-button">Explore Full Heatmap Analysis</a>
        </div>
        </div>
        """
    
    # Anomaly Detection Section
    if include_anomalies and days_with_data > 3:  # Only if we have enough data
        # Get time columns
        time_cols = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}" 
                     for hour in range(24) for i in range(2)]
        
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
                <p>The following significant changes in energy usage were detected (note significant decreases could be caused by missing data from our supplier):</p>
                
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
        
        # Standard recommendations
        html_content += """
                <li><strong>Room Management</strong>: Group bookings by floor/wing and implement vacancy setbacks (18°C winter, 24°C summer).</li>
                <li><strong>Lighting Optimization</strong>: Replace failed bulbs with LEDs and ensure lights are off in unoccupied areas.</li>
                <li><strong>Equipment Maintenance</strong>: Check HVAC filters and calibrate thermostats monthly.</li>
        """
        
        # Facility-specific recommendations
        if hotel_facilities.get('has_restaurant', False):
            html_content += """
                <li><strong>Kitchen Operations</strong>: Check refrigeration seals and train staff on equipment usage timing.</li>
            """
        
        if hotel_facilities.get('has_conf_rooms', False):
            html_content += """
                <li><strong>Conference Facilities</strong>: Implement occupancy-based HVAC control for meeting rooms.</li>
            """
        
        if hotel_facilities.get('has_pool', False):
            html_content += """
                <li><strong>Pool Management</strong>: Reduce pool temperature by 1°C and use pool covers overnight.</li>
            """
        
        html_content += """
            </ol>
            
            <div class="info">
                <p>Implementing these recommendations could save approximately 5-10% on your monthly energy costs.</p>
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
            <a href="https://4cgroup-sustainability-dashboard.streamlit.app/Electricity_Dashboard" class="cta-button">Take Action Now</a>
        </div>
    </div>
    """
    
    # Footer
    dashboard_url = "https://4cgroup-sustainability-dashboard.streamlit.app/Electricity_Dashboard"
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
            "project_id": st.secrets["gcp_service_account-elec"]["gcp_project_id"],
            "private_key_id": st.secrets["gcp_service_account-elec"]["gcp_private_key_id"],
            "private_key": st.secrets["gcp_service_account-elec"]["gcp_private_key"],
            "client_email": st.secrets["gcp_service_account-elec"]["gcp_client_email"],
            "client_id": st.secrets["gcp_service_account-elec"]["gcp_client_id"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": st.secrets["gcp_service_account-elec"]["gcp_client_x509_cert_url"]
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
            spreadsheet = client.open("Energy Dashboard Data")
        except gspread.exceptions.SpreadsheetNotFound:
            spreadsheet = client.create("Energy Dashboard Data")
            # Make it accessible to anyone with the link
            spreadsheet.share(None, perm_type='anyone', role='reader')
        
        # Check if notes worksheet exists, create if not
        try:
            worksheet = spreadsheet.worksheet("energy_notes")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title="energy_notes", rows=1000, cols=7)
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
            if record['hotel'] == hotel and record['energy_type'] == 'electricity':  
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


# Simplified Energy Forecasting & ROI Module for Electricity Dashboard
# Add these functions to your existing dashboard code

def load_occupancy_data():
    """Load occupancy data from CSV - CLEAN VERSION"""
    try:
        occupancy_df = pd.read_csv('data/occ_sleepers.csv')
        
        # Clean data
        occupancy_df['Hotel'] = occupancy_df['Hotel'].str.strip()
        occupancy_df['Occupancy Rate'] = occupancy_df['Occupancy Rate'].astype(str).str.replace('%', '').astype(float)
        
        # Parse date (DD/MM/YYYY format)
        occupancy_df['Date'] = pd.to_datetime(occupancy_df['Month'], format='%d/%m/%Y')
        occupancy_df['Month_Num'] = occupancy_df['Date'].dt.month
        occupancy_df['Year'] = occupancy_df['Date'].dt.year
        
        return occupancy_df
    except Exception as e:
        st.error(f"Error loading occupancy data: {str(e)}")
        return pd.DataFrame()

def get_historical_occupancy(occupancy_df, hotel, month):
    """Get historical occupancy - NO DEBUG OUTPUT"""
    try:
        hotel_data = occupancy_df[
            (occupancy_df['Hotel'] == hotel) & 
            (occupancy_df['Month_Num'] == month)
        ]
        
        if not hotel_data.empty:
            return hotel_data['Occupancy Rate'].mean()
        else:
            # Fallback to month average across all hotels
            month_data = occupancy_df[occupancy_df['Month_Num'] == month]
            if not month_data.empty:
                return month_data['Occupancy Rate'].mean()
            else:
                return 70.0
    except Exception as e:
        return 70.0

def prepare_ml_features(hotel_data, occupancy_df, selected_hotel):
    """Prepare features for machine learning model"""
    try:
        # Merge hotel data with occupancy data
        monthly_usage = hotel_data.groupby(['Year', 'Month']).agg({
            'Total Usage': 'sum',
            'Date': 'count'
        }).reset_index()
        monthly_usage['Daily_Avg'] = monthly_usage['Total Usage'] / monthly_usage['Date']
        
        # Add occupancy data
        monthly_usage['Occupancy'] = monthly_usage['Month'].apply(
            lambda month: get_historical_occupancy(occupancy_df, selected_hotel, month)
        )
        
        # Create additional features
        monthly_usage['Month_Sin'] = np.sin(2 * np.pi * monthly_usage['Month'] / 12)
        monthly_usage['Month_Cos'] = np.cos(2 * np.pi * monthly_usage['Month'] / 12)
        monthly_usage['Year_Normalized'] = (monthly_usage['Year'] - monthly_usage['Year'].min()) / (monthly_usage['Year'].max() - monthly_usage['Year'].min())
        
        # Seasonal indicators
        monthly_usage['Is_Winter'] = monthly_usage['Month'].isin([12, 1, 2]).astype(int)
        monthly_usage['Is_Summer'] = monthly_usage['Month'].isin([6, 7, 8]).astype(int)
        monthly_usage['Is_Shoulder'] = (~monthly_usage['Is_Winter'].astype(bool) & ~monthly_usage['Is_Summer'].astype(bool)).astype(int)
        
        return monthly_usage
    except Exception as e:
        st.error(f"Error preparing ML features: {str(e)}")
        return None

def train_ml_model(feature_data):
    """Train machine learning model for energy forecasting"""
    try:
        if not ML_AVAILABLE or feature_data is None or len(feature_data) < 10:
            return None
        
        # Prepare features and target
        feature_cols = ['Occupancy', 'Month_Sin', 'Month_Cos', 'Year_Normalized', 'Is_Winter', 'Is_Summer', 'Is_Shoulder']
        
        # Check if we have all required columns
        missing_cols = [col for col in feature_cols if col not in feature_data.columns]
        if missing_cols:
            st.warning(f"Missing columns for ML model: {missing_cols}")
            return None
        
        X = feature_data[feature_cols].fillna(0)
        y = feature_data['Daily_Avg'].fillna(feature_data['Daily_Avg'].mean())
        
        # Split data if we have enough samples
        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model (industry standard for energy forecasting)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=3,
            min_samples_leaf=2
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
        
        return {
            'model': rf_model,
            'scaler': scaler,
            'mae': mae,
            'r2': r2,
            'feature_importance': feature_importance,
            'feature_cols': feature_cols
        }
        
    except Exception as e:
        st.error(f"Error training ML model: {str(e)}")
        return None

def calculate_enhanced_baseline(hotel_data, occupancy_df, selected_hotel):
    """Enhanced baseline with ML model and traditional methods"""
    try:
        # Prepare data for ML
        feature_data = prepare_ml_features(hotel_data, occupancy_df, selected_hotel)
        
        # Train ML model
        ml_model = None
        if ML_AVAILABLE and feature_data is not None:
            ml_model = train_ml_model(feature_data)
        
        # Traditional statistical baseline
        monthly_usage = hotel_data.groupby(['Year', 'Month']).agg({
            'Total Usage': 'sum',
            'Date': 'count'
        }).reset_index()
        monthly_usage['Daily_Avg'] = monthly_usage['Total Usage'] / monthly_usage['Date']
        
        # Add occupancy data
        monthly_usage['Occupancy'] = monthly_usage['Month'].apply(
            lambda month: get_historical_occupancy(occupancy_df, selected_hotel, month)
        )
        
        # Calculate correlation
        if len(monthly_usage) > 6:
            correlation = np.corrcoef(monthly_usage['Occupancy'], monthly_usage['Daily_Avg'])[0,1]
            if np.isnan(correlation):
                correlation = 0.4  # Default for hotels
        else:
            correlation = 0.4
        
        # Enhanced seasonal factors (UK climate + AC consideration)
        overall_avg = monthly_usage['Daily_Avg'].mean()
        uk_climate_factors = {
            1: 1.25,   # January - heating peak
            2: 1.20,   # February - heating
            3: 1.10,   # March - transitional
            4: 0.95,   # April - mild
            5: 0.85,   # May - mild
            6: 1.00,   # June - AC starts
            7: 1.15,   # July - AC peak (CORRECTED)
            8: 1.15,   # August - AC peak (CORRECTED)
            9: 0.95,   # September - mild
            10: 0.90,  # October - mild
            11: 1.05,  # November - heating starts
            12: 1.20   # December - heating
        }
        
        seasonal_factors = {}
        for month in range(1, 13):
            month_data = monthly_usage[monthly_usage['Month'] == month]
            if not month_data.empty and overall_avg > 0:
                actual_factor = month_data['Daily_Avg'].mean() / overall_avg
                # Blend actual data with UK climate baseline
                seasonal_factors[month] = actual_factor * 0.6 + uk_climate_factors[month] * 0.4
            else:
                seasonal_factors[month] = uk_climate_factors[month]
        
        return {
            'baseline_daily_avg': overall_avg,
            'seasonal_factors': seasonal_factors,
            'occupancy_correlation': correlation,
            'monthly_data': monthly_usage,
            'ml_model': ml_model,
            'feature_data': feature_data
        }
        
    except Exception as e:
        st.error(f"Error calculating baseline: {str(e)}")
        return None

def ml_enhanced_forecast(baseline_data, occupancy_df, hotel, target_month, target_year, days_in_period=30):
    """ML-enhanced forecasting with fallback to statistical methods"""
    try:
        if not baseline_data:
            return None
        
        target_occupancy = get_historical_occupancy(occupancy_df, hotel, target_month)
        
        # Try ML prediction first
        ml_prediction = None
        if baseline_data.get('ml_model') and ML_AVAILABLE:
            try:
                ml_model_data = baseline_data['ml_model']
                
                # Prepare features for prediction
                month_sin = np.sin(2 * np.pi * target_month / 12)
                month_cos = np.cos(2 * np.pi * target_month / 12)
                year_norm = (target_year - 2023) / 2  # Normalize around current years
                is_winter = 1 if target_month in [12, 1, 2] else 0
                is_summer = 1 if target_month in [6, 7, 8] else 0
                is_shoulder = 1 if not (is_winter or is_summer) else 0
                
                features = np.array([[
                    target_occupancy, month_sin, month_cos, year_norm,
                    is_winter, is_summer, is_shoulder
                ]])
                
                # Scale features and predict
                features_scaled = ml_model_data['scaler'].transform(features)
                ml_prediction = ml_model_data['model'].predict(features_scaled)[0]
                
            except Exception as e:
                st.warning(f"ML prediction failed, using statistical method: {str(e)}")
        
        # Traditional statistical prediction (fallback)
        base_daily = baseline_data['baseline_daily_avg']
        seasonal_factor = baseline_data['seasonal_factors'].get(target_month, 1.0)
        
        # Non-linear occupancy modeling (research-based)
        baseline_occupancy = 70
        correlation = baseline_data['occupancy_correlation']
        
        # Diminishing returns above 80% occupancy
        if target_occupancy <= 80:
            occupancy_impact = (target_occupancy - baseline_occupancy) * correlation * 0.01
        else:
            base_impact = (80 - baseline_occupancy) * correlation * 0.01
            excess_impact = (target_occupancy - 80) * correlation * 0.005
            occupancy_impact = base_impact + excess_impact
        
        occupancy_factor = 1 + occupancy_impact
        statistical_prediction = base_daily * seasonal_factor * occupancy_factor
        
        # Combine predictions if ML is available
        if ml_prediction is not None:
            # Weight: 60% ML, 40% statistical (if ML model has good performance)
            ml_weight = 0.6 if baseline_data['ml_model']['r2'] > 0.5 else 0.3
            final_prediction = (ml_prediction * ml_weight) + (statistical_prediction * (1 - ml_weight))
            prediction_method = f"Hybrid (ML: {ml_weight:.0%}, Statistical: {1-ml_weight:.0%})"
        else:
            final_prediction = statistical_prediction
            prediction_method = "Statistical"
        
        forecasted_total = final_prediction * days_in_period
        
        return {
            'daily_forecast': final_prediction,
            'period_forecast': forecasted_total,
            'target_occupancy': target_occupancy,
            'seasonal_factor': seasonal_factor,
            'occupancy_factor': occupancy_factor,
            'correlation': correlation,
            'prediction_method': prediction_method,
            'ml_prediction': ml_prediction,
            'statistical_prediction': statistical_prediction
        }
        
    except Exception as e:
        st.error(f"Error in ML forecasting: {str(e)}")
        return None

def enhanced_roi_calculation(baseline_data, occupancy_df, hotel, implementation_date, savings_percentage, investment_cost):
    """Enhanced ROI with variable savings by season/occupancy"""
    try:
        impl_date = pd.to_datetime(implementation_date)
        months_forecast = []
        current_date = impl_date
        
        for i in range(12):
            month = current_date.month
            year = current_date.year
            days_in_month = calendar.monthrange(year, month)[1]
            
            # Get baseline forecast
            baseline_forecast = ml_enhanced_forecast(
                baseline_data, occupancy_df, hotel, month, year, days_in_month
            )
            
            if baseline_forecast:
                baseline_usage = baseline_forecast['period_forecast']
                baseline_cost = baseline_usage * 0.25
                
                # Variable savings based on occupancy and season
                occupancy = baseline_forecast['target_occupancy']
                seasonal_factor = baseline_forecast['seasonal_factor']
                
                # Higher occupancy = more equipment running = higher absolute savings
                occupancy_multiplier = 0.8 + (occupancy / 100) * 0.4  # 0.8 to 1.2 range
                
                # Peak seasons have higher savings potential
                seasonal_multiplier = 0.9 + (seasonal_factor - 1.0) * 0.2
                seasonal_multiplier = max(0.8, min(1.3, seasonal_multiplier))  # Cap between 0.8-1.3
                
                # Calculate effective savings
                effective_savings_pct = savings_percentage * occupancy_multiplier * seasonal_multiplier
                effective_savings_pct = min(effective_savings_pct, savings_percentage * 1.4)  # Cap at 40% boost
                
                saved_usage = baseline_usage * (effective_savings_pct / 100)
                saved_cost = saved_usage * 0.25
                
                months_forecast.append({
                    'Date': current_date.strftime('%Y-%m'),
                    'Month': calendar.month_name[month],
                    'Year': year,
                    'Baseline_Usage': baseline_usage,
                    'Baseline_Cost': baseline_cost,
                    'Saved_Usage': saved_usage,
                    'Saved_Cost': saved_cost,
                    'New_Usage': baseline_usage - saved_usage,
                    'New_Cost': baseline_cost - saved_cost,
                    'Occupancy': occupancy,
                    'Effective_Savings_Pct': effective_savings_pct
                })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        # Calculate payback
        cumulative_savings = 0
        payback_month = None
        
        for i, month_data in enumerate(months_forecast):
            cumulative_savings += month_data['Saved_Cost']
            month_data['Cumulative_Savings'] = cumulative_savings
            
            if payback_month is None and cumulative_savings >= investment_cost:
                payback_month = i + 1
        
        total_annual_savings = sum([m['Saved_Cost'] for m in months_forecast])
        
        return {
            'months_forecast': months_forecast,
            'total_annual_savings': total_annual_savings,
            'payback_months': payback_month,
            'average_effective_savings': np.mean([m['Effective_Savings_Pct'] for m in months_forecast])
        }
        
    except Exception as e:
        st.error(f"Error calculating ROI: {str(e)}")
        return None

def create_enhanced_chart(roi_analysis, base_savings_pct):
    """Create ROI chart with variable savings"""
    months = [month_data['Date'] for month_data in roi_analysis['months_forecast']]
    baseline_costs = [month_data['Baseline_Cost'] for month_data in roi_analysis['months_forecast']]
    new_costs = [month_data['New_Cost'] for month_data in roi_analysis['months_forecast']]
    cumulative_savings = [month_data['Cumulative_Savings'] for month_data in roi_analysis['months_forecast']]
    effective_savings = [month_data['Effective_Savings_Pct'] for month_data in roi_analysis['months_forecast']]
    
    fig = go.Figure()
    
    # Baseline costs
    fig.add_trace(go.Bar(
        x=months,
        y=baseline_costs,
        name='Baseline Cost',
        marker_color='#FF6B6B',
        opacity=0.7
    ))
    
    # New costs after investment
    fig.add_trace(go.Bar(
        x=months,
        y=new_costs,
        name='Cost After Investment',
        marker_color='#4ECDC4',
        opacity=0.8
    ))
    
    # Cumulative savings line
    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative_savings,
        mode='lines+markers',
        name='Cumulative Savings',
        line=dict(color='#45B7D1', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Enhanced ROI Analysis: Variable Savings by Month",
        xaxis_title="Month",
        yaxis_title="Monthly Cost (£)",
        yaxis2=dict(
            title="Cumulative Savings (£)",
            overlaying='y',
            side='right'
        ),
        barmode='group',
        showlegend=True,
        hovermode='x unified',
        annotations=[
            dict(
                text=f"Savings vary by occupancy & season<br>Base: {base_savings_pct}% | Range: {min(effective_savings):.1f}%-{max(effective_savings):.1f}%",
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                showarrow=False,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            )
        ]
    )
    
    return fig

# ADD THESE FUNCTIONS TO YOUR ELECTRICITY DASHBOARD CODE
# Place them BEFORE your existing show_forecasting_tab function

def analyze_actual_usage_patterns(hotel_data, selected_hotel):
    """Analyze actual usage patterns from your electricity data"""
    try:
        # Filter data for the selected hotel
        hotel_usage = hotel_data[hotel_data["Hotel"] == selected_hotel].copy()
        
        if hotel_usage.empty:
            return None
            
        # Group by year and month to get monthly totals
        monthly_analysis = hotel_usage.groupby(['Year', 'Month']).agg({
            'Total Usage': 'sum',
            'Date': 'count'  # Count of days with data
        }).reset_index()
        
        # Calculate daily averages
        monthly_analysis['Daily_Avg'] = monthly_analysis['Total Usage'] / monthly_analysis['Date']
        monthly_analysis['Month_Name'] = monthly_analysis['Month'].apply(lambda x: calendar.month_name[x])
        
        # Calculate seasonal patterns from actual data
        seasonal_factors = {}
        overall_daily_avg = monthly_analysis['Daily_Avg'].mean()
        
        for month in range(1, 13):
            month_data = monthly_analysis[monthly_analysis['Month'] == month]
            if not month_data.empty:
                month_avg = month_data['Daily_Avg'].mean()
                seasonal_factors[month] = month_avg / overall_daily_avg if overall_daily_avg > 0 else 1.0
            else:
                # Use industry defaults for missing months
                uk_defaults = {1: 1.25, 2: 1.20, 3: 1.10, 4: 0.95, 5: 0.85, 6: 1.00,
                              7: 1.15, 8: 1.15, 9: 0.95, 10: 0.90, 11: 1.05, 12: 1.20}
                seasonal_factors[month] = uk_defaults[month]
        
        # Calculate year-over-year trends
        yoy_trends = {}
        years = sorted(monthly_analysis['Year'].unique())
        
        if len(years) > 1:
            for i in range(1, len(years)):
                current_year = years[i]
                prev_year = years[i-1]
                
                current_total = monthly_analysis[monthly_analysis['Year'] == current_year]['Daily_Avg'].mean()
                prev_total = monthly_analysis[monthly_analysis['Year'] == prev_year]['Daily_Avg'].mean()
                
                if prev_total > 0:
                    yoy_change = ((current_total - prev_total) / prev_total) * 100
                    yoy_trends[f"{prev_year}-{current_year}"] = yoy_change
        
        # Calculate weekly patterns from half-hourly data
        time_cols = [f"{str(hour).zfill(2)}:{'00' if i == 0 else '30'}" 
                    for hour in range(24) for i in range(2)]
        
        # Get weekday vs weekend patterns
        hotel_usage['Day_of_Week'] = hotel_usage['Date'].dt.day_name()
        hotel_usage['Is_Weekend'] = hotel_usage['Date'].dt.dayofweek >= 5
        
        weekday_avg = hotel_usage[~hotel_usage['Is_Weekend']]['Total Usage'].mean()
        weekend_avg = hotel_usage[hotel_usage['Is_Weekend']]['Total Usage'].mean()
        
        return {
            'monthly_data': monthly_analysis,
            'seasonal_factors': seasonal_factors,
            'overall_daily_avg': overall_daily_avg,
            'yoy_trends': yoy_trends,
            'weekday_avg': weekday_avg,
            'weekend_avg': weekend_avg,
            'weekend_factor': weekend_avg / weekday_avg if weekday_avg > 0 else 1.0,
            'data_years': years,
            'total_months_data': len(monthly_analysis)
        }
        
    except Exception as e:
        st.error(f"Error analyzing usage patterns: {str(e)}")
        return None

def enhanced_forecast_with_actual_data(usage_analysis, target_month, target_year, days_in_period=30):
    """Enhanced forecasting using actual electricity data patterns"""
    try:
        if not usage_analysis:
            return None
            
        # Base forecast on actual seasonal patterns
        seasonal_factor = usage_analysis['seasonal_factors'].get(target_month, 1.0)
        base_daily_avg = usage_analysis['overall_daily_avg']
        
        # Apply year-over-year trend if available
        trend_factor = 1.0
        if usage_analysis['yoy_trends']:
            # Use average YoY trend
            avg_trend = np.mean(list(usage_analysis['yoy_trends'].values()))
            # Convert percentage to factor and apply conservatively
            trend_factor = 1 + (avg_trend / 100 * 0.5)  # Apply 50% of historical trend
        
        # Calculate forecasted daily usage
        forecasted_daily = base_daily_avg * seasonal_factor * trend_factor
        
        # Calculate total for period
        forecasted_total = forecasted_daily * days_in_period
        
        # Calculate confidence intervals based on historical variance
        monthly_data = usage_analysis['monthly_data']
        if len(monthly_data) > 3:
            # Calculate standard deviation of daily averages
            std_dev = monthly_data['Daily_Avg'].std()
            confidence_interval = std_dev * 1.96  # 95% confidence interval
            
            lower_bound = max(0, (forecasted_daily - confidence_interval) * days_in_period)
            upper_bound = (forecasted_daily + confidence_interval) * days_in_period
        else:
            # Use industry standard ±15% for new properties
            lower_bound = forecasted_total * 0.85
            upper_bound = forecasted_total * 1.15
        
        return {
            'daily_forecast': forecasted_daily,
            'period_forecast': forecasted_total,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'seasonal_factor': seasonal_factor,
            'trend_factor': trend_factor,
            'base_daily_avg': base_daily_avg,
            'confidence_range': f"±{((upper_bound - lower_bound) / 2 / forecasted_total * 100):.1f}%"
        }
        
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None

def calculate_savings_scenarios(baseline_forecast, intervention_details):
    """Calculate different savings scenarios based on intervention type"""
    
    # Define savings potential by intervention type with explanations
    intervention_library = {
        "LED Lighting Upgrade": {
            "base_savings": 8,
            "max_savings": 12,
            "description": "Replace all lighting with LED fixtures",
            "explanation": "LEDs use 50-80% less energy than traditional lighting. Savings depend on current lighting technology and operating hours.",
            "implementation_time": "1-3 months",
            "factors": {"high_operating_hours": 1.2, "old_technology": 1.4}
        },
        "Smart HVAC Controls": {
            "base_savings": 15,
            "max_savings": 25,
            "description": "Install smart thermostats and zone controls",
            "explanation": "Optimizes heating/cooling based on occupancy and schedules. Higher savings in properties with long unoccupied periods.",
            "implementation_time": "2-4 months",
            "factors": {"variable_occupancy": 1.3, "old_hvac": 1.2}
        },
        "Building Management System (BMS)": {
            "base_savings": 12,
            "max_savings": 20,
            "description": "Centralized energy management system",
            "explanation": "Coordinates all building systems for optimal efficiency. Best for larger properties with multiple energy systems.",
            "implementation_time": "3-6 months",
            "factors": {"large_property": 1.4, "complex_systems": 1.2}
        },
        "Insulation & Weatherproofing": {
            "base_savings": 10,
            "max_savings": 18,
            "description": "Improve building envelope efficiency",
            "explanation": "Reduces heating and cooling loads. More effective in older buildings with poor insulation.",
            "implementation_time": "1-2 months",
            "factors": {"old_building": 1.5, "extreme_weather": 1.2}
        },
        "Energy Management Training": {
            "base_savings": 5,
            "max_savings": 10,
            "description": "Staff training and behavioral changes",
            "explanation": "Low-cost intervention focusing on operational efficiency. Quick to implement with immediate results.",
            "implementation_time": "2-4 weeks",
            "factors": {"engaged_staff": 1.3, "clear_procedures": 1.2}
        },
        "Heat Recovery Systems": {
            "base_savings": 8,
            "max_savings": 15,
            "description": "Recover waste heat from ventilation",
            "explanation": "Captures heat from exhaust air to warm incoming fresh air. Most effective in cold climates.",
            "implementation_time": "2-3 months",
            "factors": {"cold_climate": 1.4, "high_ventilation": 1.2}
        }
    }
    
    intervention = intervention_library.get(intervention_details['type'], {
        "base_savings": intervention_details.get('custom_savings', 10),
        "max_savings": intervention_details.get('custom_savings', 10) * 1.3,
        "description": "Custom intervention",
        "explanation": "Custom energy efficiency measure",
        "implementation_time": "Variable",
        "factors": {}
    })
    
    # Calculate effective savings based on property characteristics
    base_savings = intervention["base_savings"]
    
    # Apply property-specific factors
    effective_savings = base_savings
    applied_factors = []
    
    for factor, multiplier in intervention["factors"].items():
        if intervention_details.get('property_factors', {}).get(factor, False):
            effective_savings *= multiplier
            applied_factors.append(f"{factor}: +{(multiplier-1)*100:.0f}%")
    
    # Cap at maximum savings
    effective_savings = min(effective_savings, intervention["max_savings"])
    
    # Calculate monthly savings
    baseline_usage = baseline_forecast['period_forecast']
    baseline_cost = baseline_usage * 0.25  # £0.25/kWh
    
    saved_usage = baseline_usage * (effective_savings / 100)
    monthly_savings = saved_usage * 0.25
    
    return {
        'intervention': intervention,
        'effective_savings_percent': effective_savings,
        'baseline_cost': baseline_cost,
        'monthly_savings': monthly_savings,
        'saved_usage': saved_usage,
        'new_monthly_cost': baseline_cost - monthly_savings,
        'applied_factors': applied_factors
    }

def comprehensive_roi_analysis(usage_analysis, intervention_details, implementation_date, investment_cost):
    """Comprehensive ROI analysis with seasonal variations"""
    try:
        impl_date = pd.to_datetime(implementation_date)
        analysis_months = []
        
        # Analyze 24 months to show full cycle
        current_date = impl_date
        cumulative_savings = 0
        payback_achieved = False
        payback_month = None
        
        for month_num in range(24):
            month = current_date.month
            year = current_date.year
            days_in_month = calendar.monthrange(year, month)[1]
            
            # Get baseline forecast for this month
            baseline_forecast = enhanced_forecast_with_actual_data(
                usage_analysis, month, year, days_in_month
            )
            
            if baseline_forecast:
                # Calculate savings for this month
                savings_analysis = calculate_savings_scenarios(baseline_forecast, intervention_details)
                
                # Add seasonal variation to savings (some interventions more effective in winter/summer)
                seasonal_effectiveness = 1.0
                if intervention_details['type'] in ['Smart HVAC Controls', 'Insulation & Weatherproofing']:
                    # More effective in extreme weather months
                    if month in [12, 1, 2, 6, 7, 8]:  # Winter and summer
                        seasonal_effectiveness = 1.1
                elif intervention_details['type'] == 'LED Lighting Upgrade':
                    # More effective in winter (longer dark hours)
                    if month in [10, 11, 12, 1, 2, 3]:
                        seasonal_effectiveness = 1.05
                
                adjusted_savings = savings_analysis['monthly_savings'] * seasonal_effectiveness
                cumulative_savings += adjusted_savings
                
                # Check for payback
                if not payback_achieved and cumulative_savings >= investment_cost:
                    payback_achieved = True
                    payback_month = month_num + 1
                
                analysis_months.append({
                    'Month_Num': month_num + 1,
                    'Date': current_date.strftime('%b %Y'),
                    'Month': calendar.month_name[month],
                    'Year': year,
                    'Baseline_Usage': baseline_forecast['period_forecast'],
                    'Baseline_Cost': savings_analysis['baseline_cost'],
                    'Monthly_Savings': adjusted_savings,
                    'Cumulative_Savings': cumulative_savings,
                    'Saved_Usage': savings_analysis['saved_usage'] * seasonal_effectiveness,
                    'ROI_to_Date': ((cumulative_savings - investment_cost) / investment_cost * 100) if investment_cost > 0 else 0,
                    'Payback_Achieved': payback_achieved
                })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        # Calculate summary metrics
        first_year_savings = sum([m['Monthly_Savings'] for m in analysis_months[:12]])
        second_year_savings = sum([m['Monthly_Savings'] for m in analysis_months[12:24]])
        
        return {
            'monthly_analysis': analysis_months,
            'payback_months': payback_month,
            'first_year_savings': first_year_savings,
            'second_year_savings': second_year_savings,
            'two_year_roi': ((first_year_savings + second_year_savings - investment_cost) / investment_cost * 100) if investment_cost > 0 else 0,
            'break_even_month': payback_month
        }
        
    except Exception as e:
        st.error(f"Error in ROI analysis: {str(e)}")
        return None

def create_comprehensive_roi_chart(roi_analysis, investment_cost, intervention_name):
    """Create comprehensive ROI visualization"""
    months = [m['Date'] for m in roi_analysis['monthly_analysis']]
    monthly_savings = [m['Monthly_Savings'] for m in roi_analysis['monthly_analysis']]
    cumulative_savings = [m['Cumulative_Savings'] for m in roi_analysis['monthly_analysis']]
    roi_to_date = [m['ROI_to_Date'] for m in roi_analysis['monthly_analysis']]
    
    # Create subplot figure
    fig = go.Figure()
    
    # Monthly savings bars
    fig.add_trace(go.Bar(
        x=months,
        y=monthly_savings,
        name='Monthly Savings',
        marker_color='#4ECDC4',
        opacity=0.7,
        yaxis='y'
    ))
    
    # Cumulative savings line
    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative_savings,
        mode='lines+markers',
        name='Cumulative Savings',
        line=dict(color='#45B7D1', width=3),
        yaxis='y'
    ))
    
    # Investment cost line
    fig.add_trace(go.Scatter(
        x=months,
        y=[investment_cost] * len(months),
        mode='lines',
        name='Investment Cost',
        line=dict(color='#FF6B6B', dash='dash', width=2),
        yaxis='y'
    ))
    
    # ROI percentage line
    fig.add_trace(go.Scatter(
        x=months,
        y=roi_to_date,
        mode='lines+markers',
        name='ROI (%)',
        line=dict(color='#9B59B6', width=2),
        yaxis='y2'
    ))
    
    # Add payback point annotation
    if roi_analysis['payback_months']:
        payback_month = roi_analysis['monthly_analysis'][roi_analysis['payback_months'] - 1]
        fig.add_annotation(
            x=payback_month['Date'],
            y=payback_month['Cumulative_Savings'],
            text=f"Payback Achieved!<br>Month {roi_analysis['payback_months']}",
            showarrow=True,
            arrowhead=2,
            arrowcolor='green',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='green'
        )
    
    fig.update_layout(
        title=f"ROI Analysis: {intervention_name} (24-Month View)",
        xaxis_title="Month",
        yaxis=dict(title="Savings (£)", side='left'),
        yaxis2=dict(title="ROI (%)", overlaying='y', side='right'),
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    
    return fig
# ADD THIS FUNCTION to your helper functions (before show_forecasting_tab)

def enhanced_forecast_with_regression_option(usage_analysis, target_month, target_year, days_in_period=30, use_regression=False):
    """Enhanced forecasting with optional linear regression"""
    try:
        if not usage_analysis:
            return None
            
        # Base statistical method (existing)
        seasonal_factor = usage_analysis['seasonal_factors'].get(target_month, 1.0)
        base_daily_avg = usage_analysis['overall_daily_avg']
        
        # Apply year-over-year trend
        trend_factor = 1.0
        if usage_analysis['yoy_trends']:
            avg_trend = np.mean(list(usage_analysis['yoy_trends'].values()))
            trend_factor = 1 + (avg_trend / 100 * 0.5)
        
        statistical_forecast = base_daily_avg * seasonal_factor * trend_factor
        
        # LINEAR REGRESSION FORECAST (NEW!)
        regression_forecast = statistical_forecast  # fallback
        regression_r2 = 0
        
        if use_regression and len(usage_analysis['monthly_data']) >= 6:
            try:
                monthly_data = usage_analysis['monthly_data'].copy()
                
                # Prepare features for regression
                monthly_data['Month_Sin'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
                monthly_data['Month_Cos'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)
                monthly_data['Time_Index'] = range(len(monthly_data))
                
                # Features: seasonality + time trend
                X = monthly_data[['Month_Sin', 'Month_Cos', 'Time_Index']].values
                y = monthly_data['Daily_Avg'].values
                
                # Fit linear regression
                try:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    reg_model = LinearRegression()
                    reg_model.fit(X, y)
                    
                    # Calculate R² for model quality
                    y_pred = reg_model.predict(X)
                    regression_r2 = r2_score(y, y_pred)
                    
                    # Predict target month
                    target_month_sin = np.sin(2 * np.pi * target_month / 12)
                    target_month_cos = np.cos(2 * np.pi * target_month / 12)
                    
                    # Estimate time index for target year/month
                    last_time_index = monthly_data['Time_Index'].max()
                    months_ahead = (target_year - monthly_data['Year'].max()) * 12 + (target_month - monthly_data['Month'].max())
                    target_time_index = last_time_index + months_ahead
                    
                    target_features = np.array([[target_month_sin, target_month_cos, target_time_index]])
                    regression_forecast = reg_model.predict(target_features)[0]
                    
                    # Ensure reasonable bounds
                    regression_forecast = max(0, regression_forecast)
                    
                except ImportError:
                    # Sklearn not available, fall back to statistical
                    st.warning("📊 sklearn not available. Using statistical method.")
                    regression_forecast = statistical_forecast
                    regression_r2 = 0
                    
            except Exception as e:
                st.warning(f"Linear regression failed: {str(e)}. Using statistical method.")
                regression_forecast = statistical_forecast
                regression_r2 = 0
        
        # Choose forecast method
        if use_regression and regression_r2 > 0.3:  # Use regression if R² > 0.3
            final_forecast = regression_forecast
            method = f"Linear Regression (R² = {regression_r2:.3f})"
        else:
            final_forecast = statistical_forecast
            method = "Statistical (Seasonal + Trend)"
        
        # Calculate total for period
        forecasted_total = final_forecast * days_in_period
        
        # Calculate confidence intervals
        monthly_data = usage_analysis['monthly_data']
        if len(monthly_data) > 3:
            std_dev = monthly_data['Daily_Avg'].std()
            confidence_interval = std_dev * 1.96
            lower_bound = max(0, (final_forecast - confidence_interval) * days_in_period)
            upper_bound = (final_forecast + confidence_interval) * days_in_period
        else:
            lower_bound = forecasted_total * 0.85
            upper_bound = forecasted_total * 1.15
        
        return {
            'daily_forecast': final_forecast,
            'period_forecast': forecasted_total,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'seasonal_factor': seasonal_factor,
            'trend_factor': trend_factor,
            'base_daily_avg': base_daily_avg,
            'confidence_range': f"±{((upper_bound - lower_bound) / 2 / forecasted_total * 100):.1f}%",
            'method': method,
            'regression_r2': regression_r2,
            'statistical_forecast': statistical_forecast,
            'regression_forecast': regression_forecast if use_regression else None
        }
        
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None

# REPLACE your old show_forecasting_tab function with this:
    
# ENHANCED YEAR-OVER-YEAR TRENDS ANALYSIS
# Replace the existing YoY section in your show_forecasting_tab with this:

def calculate_enhanced_yoy_trends(usage_analysis, trend_type="calendar"):
    """Calculate enhanced year-over-year trends with calendar/financial year options"""
    try:
        monthly_data = usage_analysis['monthly_data'].copy()
        
        if len(monthly_data) < 12:
            return None, None
        
        # Sort by date
        monthly_data = monthly_data.sort_values(['Year', 'Month'])
        
        # Calculate trends based on type
        if trend_type == "calendar":
            # Calendar year: Jan-Dec
            yearly_totals = monthly_data.groupby('Year').agg({
                'Total Usage': 'sum',
                'Daily_Avg': 'mean',
                'Date': 'count'  # Number of months with data
            }).reset_index()
        else:
            # Financial year: Apr-Mar (UK standard)
            monthly_data['Financial_Year'] = monthly_data.apply(
                lambda row: row['Year'] if row['Month'] >= 4 else row['Year'] - 1, axis=1
            )
            yearly_totals = monthly_data.groupby('Financial_Year').agg({
                'Total Usage': 'sum',
                'Daily_Avg': 'mean',
                'Date': 'count'
            }).reset_index()
            yearly_totals.rename(columns={'Financial_Year': 'Year'}, inplace=True)
        
        # Only include years with at least 10 months of data
        complete_years = yearly_totals[yearly_totals['Date'] >= 7].copy()
        
        if len(complete_years) < 2:
            return None, None
        
        # Calculate year-over-year changes
        yoy_changes = []
        annual_summary = []
        
        for i in range(1, len(complete_years)):
            current_year = complete_years.iloc[i]
            previous_year = complete_years.iloc[i-1]
            
            # Usage change
            usage_change = ((current_year['Total Usage'] - previous_year['Total Usage']) / previous_year['Total Usage']) * 100
            
            # Daily average change
            daily_avg_change = ((current_year['Daily_Avg'] - previous_year['Daily_Avg']) / previous_year['Daily_Avg']) * 100
            
            # Cost impact (assuming £0.25/kWh)
            cost_change = (current_year['Total Usage'] - previous_year['Total Usage']) * 0.25
            
            period_label = f"FY{int(previous_year['Year'])}/{str(int(current_year['Year']))[-2:]}" if trend_type == "financial" else f"{int(previous_year['Year'])}-{int(current_year['Year'])}"
            
            yoy_changes.append({
                'Period': period_label,
                'Previous_Year': int(previous_year['Year']),
                'Current_Year': int(current_year['Year']),
                'Previous_Usage': previous_year['Total Usage'],
                'Current_Usage': current_year['Total Usage'],
                'Usage_Change_%': usage_change,
                'Daily_Avg_Change_%': daily_avg_change,
                'Cost_Impact_£': cost_change,
                'Direction': '📈 Increasing' if usage_change > 2 else '📉 Decreasing' if usage_change < -2 else '➡️ Stable',
                'Trend_Strength': 'Strong' if abs(usage_change) > 10 else 'Moderate' if abs(usage_change) > 5 else 'Mild'
            })
        
        # Annual summary for each complete year
        for _, year_data in complete_years.iterrows():
            year_label = f"FY{int(year_data['Year'])}/{str(int(year_data['Year']) + 1)[-2:]}" if trend_type == "financial" else str(int(year_data['Year']))
            
            annual_summary.append({
                'Year': year_label,
                'Total_Usage': year_data['Total Usage'],
                'Daily_Average': year_data['Daily_Avg'],
                'Annual_Cost': year_data['Total Usage'] * 0.25,
                'Months_Data': year_data['Date']
            })
        
        return yoy_changes, annual_summary
        
    except Exception as e:
        st.error(f"Error calculating YoY trends: {str(e)}")
        return None, None

def show_enhanced_yoy_section(usage_analysis):
    """Enhanced Year-over-Year trends section with calendar/financial year options"""
    
    if not usage_analysis or len(usage_analysis.get('data_years', [])) < 2:
        st.info("ℹ️ Need at least 2 years of data for year-over-year analysis.")
        return
    
    st.subheader("📈 Year-over-Year Performance Analysis")
    
    # Year type selection
    col1, col2 = st.columns([2, 3])
    
    with col1:
        trend_type = st.radio(
            "Analysis Period:",
            options=["calendar", "financial"],
            format_func=lambda x: "Calendar Year (Jan-Dec)" if x == "calendar" else "Financial Year (Apr-Mar)",
            help="Choose whether to analyze by calendar year or UK financial year"
        )
    
    with col2:
        st.info(f"""
        **{trend_type.title()} Year Analysis**
        {'January to December comparison' if trend_type == 'calendar' else 'April to March comparison (UK standard)'}
        """)
    
    # Calculate trends
    yoy_changes, annual_summary = calculate_enhanced_yoy_trends(usage_analysis, trend_type)
    
    if not yoy_changes or not annual_summary:
        st.warning("⚠️ Insufficient data for year-over-year analysis. Need at least 10 months per year.")
        return
    
    # Overall trend summary
    st.subheader("📊 Overall Performance Summary")
    
    # Calculate overall trends
    total_changes = [change['Usage_Change_%'] for change in yoy_changes]
    avg_annual_change = np.mean(total_changes)
    latest_change = total_changes[-1] if total_changes else 0
    
    # Cumulative change over all years
    if len(annual_summary) >= 2:
        first_year_usage = annual_summary[0]['Total_Usage']
        latest_year_usage = annual_summary[-1]['Total_Usage']
        cumulative_change = ((latest_year_usage - first_year_usage) / first_year_usage) * 100 if first_year_usage > 0 else 0
    else:
        cumulative_change = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Latest Year Change",
        f"{latest_change:+.1f}%",
        help=f"Change in {yoy_changes[-1]['Period'] if yoy_changes else 'latest period'}",
        delta_color="inverse"
    )
    
    col2.metric(
        "Average Annual Change", 
        f"{avg_annual_change:+.1f}%",
        help="Average year-over-year change",
        delta_color="inverse"
    )
    
    years_span = len(annual_summary)
    col3.metric(
        f"Cumulative Change ({years_span} years)",
        f"{cumulative_change:+.1f}%",
        help=f"Total change from {annual_summary[0]['Year']} to {annual_summary[-1]['Year']}",
        delta_color="inverse"
    )
    
    # Cost impact
    latest_cost_impact = yoy_changes[-1]['Cost_Impact_£'] if yoy_changes else 0
    col4.metric(
        "Latest Cost Impact",
        f"£{latest_cost_impact:+,.0f}",
        help="Annual cost increase/decrease vs previous year",
        delta_color="inverse"
    )
    
    # Performance interpretation
    if avg_annual_change > 5:
        st.error(f"⚠️ **Rising consumption**: Usage increasing by {avg_annual_change:.1f}% annually on average. Urgent efficiency measures needed.")
    elif avg_annual_change > 2:
        st.warning(f"📈 **Moderate increase**: Usage rising by {avg_annual_change:.1f}% annually. Consider efficiency improvements.")
    elif avg_annual_change < -5:
        st.success(f"✅ **Excellent progress**: Usage decreasing by {abs(avg_annual_change):.1f}% annually. Efficiency measures working well!")
    elif avg_annual_change < -2:
        st.success(f"👍 **Good progress**: Usage decreasing by {abs(avg_annual_change):.1f}% annually. On the right track!")
    else:
        st.info(f"➡️ **Stable consumption**: Usage relatively stable with {avg_annual_change:+.1f}% average annual change.")
    
    # Detailed year-over-year table
    st.subheader("📋 Detailed Year-over-Year Comparison")
    
    if yoy_changes:
        # Create enhanced dataframe for display
        yoy_df = pd.DataFrame(yoy_changes)
        
        # Format for display
        display_df = yoy_df[['Period', 'Previous_Usage', 'Current_Usage', 'Usage_Change_%', 'Cost_Impact_£', 'Direction', 'Trend_Strength']].copy()
        display_df.columns = ['Period', 'Previous Year (kWh)', 'Current Year (kWh)', 'Change (%)', 'Cost Impact (£)', 'Trend', 'Strength']
        
        # Format numbers
        display_df['Previous Year (kWh)'] = display_df['Previous Year (kWh)'].apply(lambda x: f"{x:,.0f}")
        display_df['Current Year (kWh)'] = display_df['Current Year (kWh)'].apply(lambda x: f"{x:,.0f}")
        display_df['Change (%)'] = display_df['Change (%)'].apply(lambda x: f"{x:+.1f}%")
        display_df['Cost Impact (£)'] = display_df['Cost Impact (£)'].apply(lambda x: f"£{x:+,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
    
    # Annual summary table
    st.subheader("📈 Annual Performance Summary")
    
    if annual_summary:
        summary_df = pd.DataFrame(annual_summary)
        
        # Format for display
        summary_df['Total Usage (kWh)'] = summary_df['Total_Usage'].apply(lambda x: f"{x:,.0f}")
        summary_df['Daily Average (kWh)'] = summary_df['Daily_Average'].apply(lambda x: f"{x:.1f}")
        summary_df['Annual Cost (£)'] = summary_df['Annual_Cost'].apply(lambda x: f"£{x:,.2f}")
        summary_df['Data Quality'] = summary_df['Months_Data'].apply(
            lambda x: f"{x}/12 months" + (" ✅" if x >= 12 else " ⚠️" if x >= 10 else " ❌")
        )
        
        display_summary = summary_df[['Year', 'Total Usage (kWh)', 'Daily Average (kWh)', 'Annual Cost (£)', 'Data Quality']]
        st.dataframe(display_summary, use_container_width=True)
    
    # Trend visualization
    st.subheader("📊 Usage Trend Visualization")
    
    if len(annual_summary) >= 2:
        fig_trend = go.Figure()
        
        # Usage trend line
        fig_trend.add_trace(go.Scatter(
            x=[summary['Year'] for summary in annual_summary],
            y=[summary['Total_Usage'] for summary in annual_summary],
            mode='lines+markers',
            name='Annual Usage',
            line=dict(color='#3366FF', width=3),
            marker=dict(size=10)
        ))
        
        # Add trend line
        years_numeric = list(range(len(annual_summary)))
        usage_values = [summary['Total_Usage'] for summary in annual_summary]
        
        # Calculate linear trend
        if len(usage_values) > 1:
            z = np.polyfit(years_numeric, usage_values, 1)
            trend_line = np.poly1d(z)
            
            fig_trend.add_trace(go.Scatter(
                x=[summary['Year'] for summary in annual_summary],
                y=trend_line(years_numeric),
                mode='lines',
                name='Trend Line',
                line=dict(color='#FF4D6D', dash='dash', width=2),
                opacity=0.8
            ))
        
        # Annotations for significant changes
        for i, change in enumerate(yoy_changes):
            if abs(change['Usage_Change_%']) > 10:  # Significant change
                fig_trend.add_annotation(
                    x=annual_summary[i+1]['Year'],
                    y=annual_summary[i+1]['Total_Usage'],
                    text=f"{change['Usage_Change_%']:+.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='red' if change['Usage_Change_%'] > 0 else 'green'
                )
        
        fig_trend.update_layout(
            title=f"Annual Usage Trend - {trend_type.title()} Year Analysis",
            xaxis_title="Year",
            yaxis_title="Usage (kWh)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Action recommendations based on trends
    st.subheader("🎯 Trend-Based Recommendations")
    
    if avg_annual_change > 5:
        st.markdown("""
        ### 🚨 **Urgent Actions Required**
        - Conduct immediate energy audit to identify causes of increase
        - Implement quick-win efficiency measures within 30 days
        - Review equipment performance and maintenance schedules
        - Consider major efficiency upgrades with ROI analysis
        - Set aggressive reduction targets for next year
        """)
    elif avg_annual_change > 2:
        st.markdown("""
        ### ⚠️ **Proactive Measures Recommended**
        - Schedule comprehensive energy assessment
        - Implement staff energy awareness training
        - Review and optimize operational schedules
        - Plan efficiency improvements for next budget cycle
        """)
    elif avg_annual_change < -2:
        st.markdown("""
        ### ✅ **Maintain and Accelerate Progress**
        - Document successful efficiency measures for replication
        - Share best practices across other properties
        - Consider additional investments with proven ROI
        - Set more ambitious targets to continue improvement
        """)
    else:
        st.markdown("""
        ### 📊 **Stable Performance - Optimization Opportunities**
        - Look for seasonal optimization opportunities
        - Benchmark against industry standards
        - Consider upgrading to more efficient technologies
        - Implement continuous improvement processes
        """)

# UPDATE your show_forecasting_tab function to include this section
# Replace the existing YoY trends section with this call:

# In your show_forecasting_tab function, replace the YoY section with:

def show_forecasting_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month):
    """Enhanced forecasting tab with linear regression options"""
    st.header("🔮 Advanced Energy Forecasting & ROI Analysis")
    
    # Explanatory header with regression info
    with st.expander("📖 Forecasting Methods Explained", expanded=False):
        st.markdown("""
        ### Available Forecasting Methods
        
        **1. Statistical Method (Default)**
        - Uses your seasonal patterns + year-over-year trends
        - Best for: Properties with clear seasonal patterns
        - Pros: Simple, interpretable, works with limited data
        
        **2. Linear Regression (Advanced)**  
        - Uses machine learning to find patterns in your data
        - Combines seasonality with time trends automatically
        - Best for: Properties with 6+ months of data
        - Pros: Can capture complex patterns, often more accurate
        
        **Which to choose?**
        - Start with Statistical for transparency
        - Try Linear Regression if you want higher accuracy
        - Compare both to see which works better for your property
        """)
    
    # Analyze patterns
    with st.spinner("Analyzing your electricity usage patterns..."):
        usage_analysis = analyze_actual_usage_patterns(hotel_data, selected_hotel)
    
    if not usage_analysis:
        st.error("❌ Unable to analyze usage patterns. Please check your data.")
        return
    
    # Data quality metrics
    st.subheader("📊 Your Data Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Data Years Available",
        f"{len(usage_analysis['data_years'])} years",
        help=f"Years: {', '.join(map(str, usage_analysis['data_years']))}"
    )
    
    col2.metric(
        "Monthly Records",  
        f"{usage_analysis['total_months_data']} months",
        help="Total months of data available for analysis"
    )
    
    col3.metric(
        "Daily Average Usage",
        f"{usage_analysis['overall_daily_avg']:,.0f} kWh",
        help="Average daily electricity usage across all available data"
    )
    
    weekend_vs_weekday = ((usage_analysis['weekend_factor'] - 1) * 100)
    col4.metric(
        "Weekend vs Weekday",
        f"{weekend_vs_weekday:+.1f}%",
        help="Weekend usage compared to weekday usage",
        delta_color="inverse"
    )
    
    # Seasonal patterns chart
    st.subheader("🌟 Your Seasonal Usage Patterns")
    
    months = [calendar.month_name[m] for m in range(1, 13)]
    factors = [usage_analysis['seasonal_factors'][m] for m in range(1, 13)]
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Bar(
        x=months,
        y=factors,
        marker_color=['#FF6B6B' if f > 1.1 else '#4ECDC4' if f < 0.9 else '#95A5A6' for f in factors],
        text=[f"{f:.2f}x" for f in factors],
        textposition='auto'
    ))
    
    fig_seasonal.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                          annotation_text="Average Usage")
    
    fig_seasonal.update_layout(
        title="Your Property's Seasonal Usage Pattern",
        xaxis_title="Month",
        yaxis_title="Usage Factor (vs Annual Average)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # show_enhanced_yoy_section(usage_analysis)

    
    # CREATE TABS
    forecast_tab, roi_tab = st.tabs(["🔮 Usage Forecasting", "💰 ROI Calculator"])
    
    with forecast_tab:
        st.subheader("🎯 Forecast Future Usage")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_year = st.selectbox(
                "Forecast Year",
                options=list(range(selected_year, selected_year + 3)),
                index=0
            )
            
        with col2:
            forecast_month = st.selectbox(
                "Forecast Month",
                options=list(range(1, 13)),
                index=selected_month - 1 if selected_month <= 12 else 0,
                format_func=lambda x: calendar.month_name[x]
            )
        
        with col3:
            # METHOD SELECTION
            use_regression = st.checkbox(
                "🤖 Use Linear Regression",
                value=False,
                help="Try machine learning for potentially more accurate forecasts"
            )
        
        # Calculate forecast with selected method
        days_in_forecast_month = calendar.monthrange(forecast_year, forecast_month)[1]
        forecast_result = enhanced_forecast_with_regression_option(
            usage_analysis, forecast_month, forecast_year, days_in_forecast_month, use_regression
        )
        
        if forecast_result:
            st.subheader(f"📊 Forecast for {calendar.month_name[forecast_month]} {forecast_year}")
            
            # Show method used
            method_color = "🤖" if "Regression" in forecast_result['method'] else "📊"
            st.info(f"{method_color} **Method**: {forecast_result['method']}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Forecasted Usage",
                f"{forecast_result['period_forecast']:,.0f} kWh",
                help="Best estimate based on your historical patterns"
            )
            
            col2.metric(
                "Estimated Cost",
                f"£{forecast_result['period_forecast'] * 0.25:,.2f}",
                help="At current electricity rates (£0.25/kWh)"
            )
            
            col3.metric(
                "Daily Average",
                f"{forecast_result['daily_forecast']:,.1f} kWh",
                help="Expected daily usage for this month"
            )
            
            col4.metric(
                "Confidence Range",
                forecast_result['confidence_range'],
                help="95% confidence interval for the forecast"
            )
            
            # Show forecast quality
            if "Regression" in forecast_result['method'] and forecast_result['regression_r2'] > 0:
                quality = 'Excellent' if forecast_result['regression_r2'] > 0.8 else 'Good' if forecast_result['regression_r2'] > 0.6 else 'Fair'
                st.success(f"✅ **Regression Quality**: R² = {forecast_result['regression_r2']:.3f} ({quality})")
    
    with roi_tab:
        st.subheader("💰 Energy Efficiency ROI Calculator")
        st.markdown("ROI analysis using your actual usage patterns.")
        
        # Simplified ROI for now
        col1, col2 = st.columns(2)
        
        with col1:
            intervention_type = st.selectbox(
                "Intervention Type",
                ["LED Lighting Upgrade", "Smart HVAC Controls", "Energy Management Training"]
            )
            investment_cost = st.number_input("Investment Cost (£)", value=10000.0)
        
        with col2:
            savings_percent = st.slider("Expected Savings (%)", 5, 25, 12)
            
        if st.button("Calculate ROI"):
            annual_baseline_cost = usage_analysis['overall_daily_avg'] * 365 * 0.25
            annual_savings = annual_baseline_cost * (savings_percent / 100)
            payback_months = investment_cost / (annual_savings / 12) if annual_savings > 0 else float('inf')
            
            col1, col2 = st.columns(2)
            col1.metric("Annual Savings", f"£{annual_savings:,.2f}")
            col2.metric("Payback Period", f"{payback_months:.1f} months" if payback_months != float('inf') else "> 60 months")

def main():
    add_dashboard_chatbot()
    client = setup_google_sheets()
    if not client:
        st.warning("Note-taking functionality may be limited due to Google Sheets configuration issues.")

    st.title("Hotels Electricity Management Dashboard")
    # Load data and initialize filters
    data = load_data()
    if data is None:
        st.stop()

        # Add this line to check for duplicates after loading (just like in gas script)
    data = check_for_duplicates(data)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_hotel = st.sidebar.selectbox(
        "Select Hotel",
        options=sorted(data["Hotel"].unique()),
        help="Choose a hotel to view its energy data",
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
        index=years.index(str(latest_year)) if str(latest_year) in years else len(years)-1
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
        key="occupancy_slider"  # Unique key for the occupancy slider
    )

    # Create tabs for different sections
# WITH this:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Overview",
        "⚡ Real-Time Management", 
        "🔮 Forecasting & ROI",  # MOVED HERE (was tab 10)
        "📈 Heat Map",
        "📅 Monthly Graph", 
        "🎯 Benchmarking & Goals",
        "📋 Action Center",
        "🎯 Targets",
        "🌍 Carbon Emissions",
        "💰 Cost Control"
    ])
    # Display content in each tab
# Display content in each tab
    with tab1:
        show_overview_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month, hotel_facilities, occupancy_rate)
        show_monthly_notes_summary(selected_hotel, selected_year, selected_month, tab_id="overview")
    with tab2:
        # Pass the filtered monthly_data, hotel_facilities, occupancy_rate, and selected year/month for anomaly detection
        show_realtime_tab(monthly_data, hotel_data, hotel_facilities, occupancy_rate, selected_year, selected_month)
        show_monthly_notes_summary(selected_hotel, selected_year, selected_month, tab_id="realtime")

    with tab3:
        show_forecasting_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month)

    with tab4:
        show_operational_tab(monthly_data, selected_hotel, selected_year, selected_month)

    with tab5:
        show_monthly_graph_tab(data, selected_hotel)

    with tab6:
        show_benchmarking_tab(data, monthly_data, selected_hotel, selected_year, selected_month)


    with tab7:
        show_action_center_tab(monthly_data, hotel_facilities, occupancy_rate)
    
    with tab8:
        show_targets_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month)
    
    with tab9:
        show_carbon_tab(monthly_data, hotel_data, selected_hotel, selected_year, selected_month)

    with tab10:
        show_cost_control_tab(monthly_data, hotel_facilities, occupancy_rate)
    
    # Add this at the end of the main function
    add_email_report_section(selected_hotel, selected_year, selected_month, data)
if __name__ == "__main__":
    main()