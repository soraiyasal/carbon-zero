# shared_components.py
import streamlit as st
import streamlit.components.v1 as components


def add_dashboard_chatbot():
    """Smart FAQ system - instant answers, no API needed"""
    
    with st.sidebar:
        with st.expander("💬 Dashboard Help", expanded=False):
            st.markdown("### Get Instant Answers")
            
            # Quick answer buttons
            help_topics = {
                "🏆 How do I improve my score?": """
                **Ways to Improve Your Score:**
                
                🎯 **Recycling (300 points):** Meet your hotel's recycling target
                ⚡ **Utilities (300 points):** Reduce water/gas/electricity per guest vs last year
                🍽️ **Food Waste (200 points):** Reduce waste per guest vs last month  
                📋 **Tasks (25 each):** Complete monthly sustainability tasks
                📅 **Meetings (25 each):** Attend sustainability meetings
                """,
                
                "📍 Where do I find my data?": """
                **Finding Your Hotel's Data:**
                
                📊 **Leaderboard Tab:** Your overall ranking and points
                🏨 **Utilities Tab:** Water, gas, electricity usage per guest
                ♻️ **Waste Tab:** Recycling rates and food waste analysis
                📄 **Paper Tab:** Select your hotel from dropdown menu
                🎯 **Task Board:** Complete monthly sustainability actions
                """,
                
                "🎯 What's my recycling target?": """
                **Recycling Targets by Hotel:**
                
                - **Camden:** 55% | **Westin:** 70% | **Canopy:** 70%
                - **CIV:** 60% | **EH:** 55% | **CIE:** 60% | **St Albans:** 55%
                
                Find your current rate in **Waste → Recycling** tab.
                """,
                
                "📊 How are points calculated?": """
                **Points System Breakdown:**
                
                **Recycling:** 300 for meeting target, 200 for close
                **Utilities:** 100 per utility (water/gas/electricity) for 10%+ reduction
                **Food Waste:** 200 for 10%+ reduction per guest vs last month
                **Paper Usage:** 200 for 10%+ reduction vs last month
                **Tasks:** 25 points each completed
                **Meetings:** 25 points each attended
                """,
                
                "👥 What does 'per guest' mean?": """
                **Per Guest Metrics Explained:**
                
                We divide total usage by number of guests to compare hotels fairly:
                - **Water per guest** = Total water ÷ Number of sleepers
                - **Energy per guest** = Total electricity ÷ Number of sleepers
                
                This lets us compare efficiency regardless of hotel size or occupancy.
                """,
                
                "📈 Why did my score change?": """
                **Score Changes Explained:**
                
                **Monthly Updates:** Scores update when new data is added
                **Comparisons:** Utilities compare to same month last year
                **Targets:** Recycling targets are specific to each hotel
                **Tasks:** New monthly tasks are added regularly
                
                Check the **Leaderboard** tab for your latest breakdown.
                """
            }
            
            # Display as clickable buttons
            for question, answer in help_topics.items():
                if st.button(question, key=f"faq_{hash(question)}", use_container_width=True):
                    st.success(answer)
            
            st.markdown("---")
            st.markdown("**Still need help?** Contact your Green Champion!")

def add_welcome_popup():
    """Show a welcome popup with help guidance - appears once per session"""
    
    # Only show once per session
    if 'welcome_shown' not in st.session_state:
        st.session_state.welcome_shown = True
        
        # Create a modal-like popup using st.info with custom styling
        st.markdown("""
        <div style="
            position: fixed;
            top: 20%;
            left: 50%;
            transform: translateX(-50%);
            z-index: 999;
            background: white;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            border: 2px solid #10b981;
            max-width: 500px;
            animation: slideIn 0.3s ease-out;
        ">
            <h2 style="color: #10b981; text-align: center; margin-top: 0;">
                👋 Welcome to Green Champions!
            </h2>
            <p style="text-align: center; font-size: 1.1rem;">
                Need help navigating the dashboard?
            </p>
            <div style="background: #f0fdf4; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                <strong>💡 Quick Tip:</strong> Look for the <strong>"💬 Dashboard Help"</strong> 
                section in the sidebar for instant answers to common questions!
            </div>
            <p style="text-align: center; margin-bottom: 0;">
                Click the help buttons for guidance on scores, data locations, and targets.
            </p>
        </div>
        
        <style>
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-50%) translateY(-20px); }
            to { opacity: 1; transform: translateX(-50%) translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Auto-hide after 5 seconds
        import time
        time.sleep(12)
        st.rerun()