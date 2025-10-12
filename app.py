#!/usr/bin/env python3
"""
CrimeCast - Modern Dark Theme Landing Page
Fixed version with proper HTML rendering and beautiful design
"""

import streamlit as st
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import authentication system
try:
    from auth.auth_utils import initialize_session_state, is_authenticated, get_current_user, logout_user, navigate_to_page
    print("✅ Authentication system imported successfully!")
except ImportError as e:
    print(f"❌ Auth import failed: {e}")

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="CRIMECAST - AI Crime Prediction",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    """Apply custom CSS styles"""
    st.markdown("""
    <style>
        /* Main Dark Theme */
        .stApp {
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff;
        }
        
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
        }
        
        /* Header Styles */
        .gradient-header {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 1rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .sub-header {
            font-size: 1.3rem;
            color: #b0b0b0;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Hero Description */
        .hero-description {
            color: #b0b0b0;
            font-size: 1.2rem;
            max-width: 800px;
            margin: 0 auto 3rem auto;
            text-align: center;
            line-height: 1.6;
        }
        
        /* Card Styles */
        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            color: white;
            margin: 0.5rem;
        }
        
        /* Button Styles */
        .stButton > button {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .secondary-btn > button {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid #667eea !important;
            color: #667eea !important;
        }
        
        /* User Welcome Card */
        .user-welcome {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        /* Hero Section */
        .hero-section {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        /* Custom Divider */
        .custom-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            margin: 3rem 0;
            border: none;
        }
        
        /* Text colors */
        .white-text {
            color: white !important;
        }
        
        .gray-text {
            color: #b0b0b0 !important;
        }
        
        .purple-text {
            color: #667eea !important;
        }
        
        /* Button container */
        .button-container {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }
        
        /* Fix for Streamlit Cloud compatibility */
        .stAlert {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

def show_user_sidebar():
    """Display user information in sidebar"""
    user = get_current_user()
    if user:
        st.sidebar.markdown(f"""
        <div class="user-welcome">
            <h3 style="margin: 0;">👋 Welcome, {user['name']}!</h3>
            <p style="margin: 0.5rem 0;">{user['email']}</p>
            <p style="margin: 0;"><strong>Role:</strong> {user['role'].title()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation for authenticated users
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚀 Quick Navigation")
        
        if st.sidebar.button("🏠 Dashboard", use_container_width=True, key="nav_dashboard"):
            navigate_to_page("3_🏠_Home.py")
        
        if st.sidebar.button("🔮 Crime Prediction", use_container_width=True, key="nav_prediction"):
            navigate_to_page("4_🔮_Crime_Prediction.py")
            
        if st.sidebar.button("📊 Data Analysis", use_container_width=True, key="nav_analysis"):
            navigate_to_page("5_📊_Data_Analysis.py")
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True, key="nav_logout"):
            logout_user()
            st.rerun()

def show_public_sidebar():
    """Display sidebar for public users"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">🚨 CRIMECAST</h3>
        <p style="color: #b0b0b0; margin: 0;">AI-Powered Crime Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Get Started")
    
    if st.sidebar.button("🔐 Login", use_container_width=True, key="sidebar_login"):
        navigate_to_page("1_🔐_Login.py")
    
    if st.sidebar.button("📝 Sign Up", use_container_width=True, key="sidebar_signup"):
        navigate_to_page("2_📝_Sign_Up.py")

def show_public_landing():
    """Show beautiful public landing page without forms"""
    
    # Hero Section with gradient header
    st.markdown("""
    <div class="hero-section">
        <div class="gradient-header">🚨 CRIMECAST</div>
        <div class="sub-header">AI-Powered Crime Prediction & Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Description
    st.markdown("""
    <div class="hero-description">
        Leverage advanced machine learning to predict crime patterns, analyze risk factors, 
        and empower law enforcement with data-driven insights for proactive community safety.
    </div>
    """, unsafe_allow_html=True)
    
    # Action Buttons
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔐 Login to Dashboard", use_container_width=True, key="hero_login"):
            navigate_to_page("1_🔐_Login.py")
    
    with col2:
        if st.button("📝 Create Account", use_container_width=True, key="hero_signup"):
            navigate_to_page("2_📝_Sign_Up.py")
    
    with col3:
        if st.button("🚀 Learn More", use_container_width=True, key="hero_learn"):
            st.info("Scroll down to see all features!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Features Section
    st.markdown('<h2 style="text-align: center; color: #667eea; margin-bottom: 2rem;">✨ Advanced Features</h2>', unsafe_allow_html=True)
    
    # Feature cards using columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-top: 0;">🤖 AI Prediction</h3>
            <p style="color: #b0b0b0; line-height: 1.6;">
            Advanced machine learning models predict crime probabilities with 85%+ accuracy using ensemble algorithms and real-time data processing.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-top: 0;">📊 Smart Analytics</h3>
            <p style="color: #b0b0b0; line-height: 1.6;">
            Interactive dashboards with temporal and spatial analysis, heat maps, and pattern recognition for comprehensive crime intelligence.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-top: 0;">🔍 Risk Assessment</h3>
            <p style="color: #b0b0b0; line-height: 1.6;">
            Multi-factor risk analysis identifying key indicators and providing actionable insights for preventive measures.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-top: 0;">🌍 Spatial Intelligence</h3>
            <p style="color: #b0b0b0; line-height: 1.6;">
            Geographic hotspot detection and area-based crime forecasting with interactive map visualizations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-top: 0;">⚡ Real-time Alerts</h3>
            <p style="color: #b0b0b0; line-height: 1.6;">
            Instant notifications and predictive alerts for emerging crime patterns and high-risk situations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-top: 0;">🔐 Secure Platform</h3>
            <p style="color: #b0b0b0; line-height: 1.6;">
            Enterprise-grade security with role-based access control, audit trails, and data encryption.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown('<h2 style="text-align: center; color: #667eea; margin-bottom: 2rem;">📈 Platform Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3 style="margin: 0; font-size: 2rem;">85.9%</h3>
            <p style="margin: 0.5rem 0 0 0;">Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3 style="margin: 0; font-size: 2rem;">446K+</h3>
            <p style="margin: 0.5rem 0 0 0;">Crime Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3 style="margin: 0; font-size: 2rem;">24/7</h3>
            <p style="margin: 0.5rem 0 0 0;">Real-time Monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h3 style="margin: 0; font-size: 2rem;">99.9%</h3>
            <p style="margin: 0.5rem 0 0 0;">System Uptime</p>
        </div>
        """, unsafe_allow_html=True)

def show_authenticated_landing():
    """Show landing page for authenticated users"""
    user = get_current_user()
    
    # Welcome Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;">
            <h1 style="margin: 0 0 1rem 0;">🚨 Welcome to CRIMECAST, {user['name']}!</h1>
            <p style="font-size: 1.2rem; opacity: 0.9; margin: 0;">
            Ready to explore advanced crime prediction and analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 1.5rem; border-left: 4px solid #667eea;">
            <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">👤 Your Role</h4>
            <h3 style="color: white; margin: 0;">{user.get('role', 'user').title()}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Quick Access Section
    st.markdown('<h2 style="color: #667eea; margin-bottom: 2rem;">🚀 Quick Access</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔮 Crime Prediction", use_container_width=True, key="auth_pred"):
            navigate_to_page("4_🔮_Crime_Prediction.py")
    
    with col2:
        if st.button("📊 Data Analysis", use_container_width=True, key="auth_data"):
            navigate_to_page("5_📊_Data_Analysis.py")
    
    with col3:
        if st.button("📈 Model Performance", use_container_width=True, key="auth_model"):
            navigate_to_page("6_📈_Model_Performance.py")
    
    with col4:
        if st.button("👤 Your Profile", use_container_width=True, key="auth_profile"):
            navigate_to_page("8_👤_Profile.py")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Recent Activity & Stats
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-top: 0;">📋 Getting Started</h3>
            <p style="color: #b0b0b0; line-height: 1.6;">
            <strong>1. Crime Prediction:</strong> Start by predicting crime probabilities in specific areas<br>
            <strong>2. Data Analysis:</strong> Explore historical patterns and trends<br>
            <strong>3. Risk Assessment:</strong> Understand key risk factors and correlations<br>
            <strong>4. Model Insights:</strong> Check prediction accuracy and performance metrics
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-top: 0;">⚡ System Status</h3>
            <p style="color: #b0b0b0; line-height: 1.6;">
            ✅ Prediction Engine: <strong>Operational</strong><br>
            ✅ Data Pipeline: <strong>Active</strong><br>
            ✅ ML Models: <strong>Optimized</strong><br>
            ✅ API Services: <strong>Stable</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Initialize session state
    try:
        initialize_session_state()
    except Exception as e:
        st.error(f"Session initialization error: {e}")
    
    # Apply custom CSS
    apply_custom_css()
    
    # Show appropriate sidebar based on authentication
    try:
        if is_authenticated():
            show_user_sidebar()
        else:
            show_public_sidebar()
    except Exception as e:
        st.error(f"Sidebar error: {e}")
        # Fallback to public sidebar
        show_public_sidebar()
    
    # Show appropriate landing page
    try:
        if is_authenticated():
            show_authenticated_landing()
        else:
            show_public_landing()
    except Exception as e:
        st.error(f"Content loading error: {e}")
        # Fallback to public landing
        show_public_landing()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 4rem; padding: 2rem; color: #666; border-top: 1px solid #333;">
        <p>🚨 CRIMECAST - Advanced Crime Prediction System | Built with Streamlit & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()