import streamlit as st


# Page configuration
st.set_page_config(
    page_title="4C Hotels Sustainability Hub",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced design
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e0eafc, #cfdef3);
    }
    
    /* Dashboard cards */
    .dashboard-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s, box-shadow 0.3s;
        cursor: pointer;
        margin: 10px 0;
    }
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.2);
    }
    .dashboard-card h3 {
        margin-bottom: 10px;
        color: #2a2a2a;
    }
    .dashboard-card ul {
        padding-left: 20px;
        color: #555555;
    }

    /* Column layout adjustments */
    .stColumn > div {
        padding: 10px;
    }
    
    /* 4C Hotel branding in sidebar */
    .hotel-branding {
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Remove Streamlit default styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Add 4C Hotels branding at the bottom of sidebar
st.sidebar.markdown("""
    <div class="hotel-branding">
        <h2>🏛️ 4C Hotels</h2>
        <p style="font-size: 1.2em;">Sustainability Hub</p>
    </div>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <h1>🏨 4C Hotels Sustainability Hub</h1>
        <p style="font-size: 1.2em; color: #555;">Explore dashboards, track progress, and drive sustainability across properties.</p>
    </div>
""", unsafe_allow_html=True)

# Dashboard cards in three columns
col1, col2, col3 = st.columns(3)

# Dashboard info
dashboard_info = [
    ("🌱 Green Champions Dashboard", [
        "Track recycling rates",
        "Monitor energy usage",
        "Analyse water consumption",
        "Manage food waste"
    ]),
    ("⚡ Electricity Dashboard", [
        "Half-hourly usage patterns",
        "Peak demand analysis",
        "Cost optimization", 
        "Anomaly detection"
    ]),
    ("🔥 Gas Dashboard", [
        "Half-hourly gas consumption",
        "Baseload analysis",
        "Cost analysis",
        "UK benchmarking"
    ]),
    ("📊 Performance Scorecard", [
        "Cost analysis",
        "Carbon footprint",
        "Business insights",
        "Portfolio comparison"
    ]),
    ("♻️ Recycling Performance", [
        "Daily recycling rates",
        "Waste composition",
        "Weekly trends",
        "Performance insights"
    ]),
    ("🎯 OKR Tracker", [
        "Department progress tracking",
        "Initiative status updates",
        "Performance monitoring",
        "Target completion dates"
    ]),
    ("🌮 LimeTrack - Food Waste", [
        "Hourly food waste tracking",
        "Waste composition analysis",
        "Cost impact assessment",
        "Reduction strategies"
    ])
]

# Render cards dynamically
columns = [col1, col2, col3]
for i, (title, details) in enumerate(dashboard_info):
    with columns[i % 3]:
        st.markdown(f"""
            <div class="dashboard-card">
                <h3>{title}</h3>
                <ul>
                    {''.join(f"<li>{item}</li>" for item in details)}
                </ul>
            </div>
        """, unsafe_allow_html=True)

# Help section
with st.expander("ℹ️ Need Help?"):
    st.markdown("""
    ### Quick Start Guide
    1. Navigate using the sidebar to explore dashboards.
    2. Filter, explore, and export data in dashboards.
    3. Contact the sustainability team for further assistance.
    """)