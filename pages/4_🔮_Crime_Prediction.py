#!/usr/bin/env python3
"""
Crime Prediction Page - Fixed for Streamlit Cloud Deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, get_current_user
from auth.decorators import login_required

# Import with error handling
try:
    from src.predictor import CrimePredictorAPI
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Predictor import failed: {e}")
    PREDICTOR_AVAILABLE = False

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

def safe_geocode_address(address, max_retries=2):
    """Convert address to latitude/longitude coordinates with retry logic"""
    if not GEOPY_AVAILABLE:
        return None, None, "Geopy not available"
        
    for attempt in range(max_retries):
        try:
            # Use different user agent for each attempt
            geolocator = Nominatim(user_agent=f"crimecast_app_{attempt}", timeout=10)
            location = geolocator.geocode(address + ", Chicago, IL, USA")
            
            if location:
                return location.latitude, location.longitude, None
            else:
                return None, None, "Address not found in Chicago area"
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            return None, None, f"Geocoding service timeout after {max_retries} attempts"
        except Exception as e:
            return None, None, f"Geocoding error: {str(e)}"

def load_models():
    """Load trained models with error handling"""
    if not PREDICTOR_AVAILABLE:
        return None
        
    try:
        # Try multiple possible model paths
        possible_paths = [
            "models/best_model_arrest.pkl",
            "./models/best_model_arrest.pkl",
            "../models/best_model_arrest.pkl",
            os.path.join(parent_dir, "models", "best_model_arrest.pkl")
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            st.error("❌ Model file not found. Please ensure models are trained and available.")
            return None
            
        predictor = CrimePredictorAPI(
            model_path=model_path,
            feature_scaler_path="models/feature_scaler.pkl",
            label_encoders_path="models/label_encoders.pkl"
        )
        return predictor
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

@st.cache_data(show_spinner=False)
def safe_load_data():
    """Load data with better error handling"""
    try:
        possible_paths = [
            'data/cleaned_crime_data.csv',
            './data/cleaned_crime_data.csv', 
            '../data/cleaned_crime_data.csv',
            os.path.join(parent_dir, 'data', 'cleaned_crime_data.csv'),
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    # Fix potential Arrow serialization issues
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            try:
                                df[col] = pd.to_numeric(df[col], errors='ignore')
                            except:
                                pass
                        # Convert int64 to int32 for Arrow compatibility
                        elif str(df[col].dtype) == 'int64':
                            df[col] = df[col].astype('int32')
                    return df
            except Exception as e:
                continue
        return None
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None

def get_feature_importance(_predictor, features):
    """Get feature importance for the prediction"""
    try:
        # Mock feature importance (replace with actual model feature importance)
        importance_data = {
            'Location Type': 0.25,
            'Time of Day': 0.20,
            'Day of Week': 0.15,
            'District': 0.12,
            'Community Area': 0.10,
            'Hour': 0.08,
            'Season': 0.05,
            'Year': 0.03,
            'Month': 0.02
        }
        
        # Adjust based on actual input values
        if features.get('TimeOfDay') == 'Night':
            importance_data['Time of Day'] += 0.10
        if features.get('Location_Description_Clean') in ['ALLEY', 'PARKING LOT']:
            importance_data['Location Type'] += 0.08
            
        return importance_data
    except Exception:
        return {
            'Location Type': 0.3,
            'Time of Day': 0.25,
            'Day of Week': 0.2,
            'District': 0.15,
            'Community Area': 0.1
        }

def get_prediction_reliability(probability):
    """Convert confidence score to police-friendly reliability rating"""
    if probability >= 0.9:
        return "🟢 HIGH RELIABILITY", "Very reliable prediction based on strong patterns"
    elif probability >= 0.7:
        return "🟡 MODERATE RELIABILITY", "Good reliability with consistent patterns"
    elif probability >= 0.5:
        return "🟠 FAIR RELIABILITY", "Moderate reliability with some uncertainty"
    else:
        return "🔴 LOW RELIABILITY", "Lower reliability - consider additional factors"

def get_operational_priority(probability, risk_level):
    """Convert to police operational priority"""
    if risk_level == "High" and probability >= 0.7:
        return "🚨 IMMEDIATE ACTION", "High probability arrest scenario - prioritize response"
    elif risk_level == "High" or probability >= 0.6:
        return "⚠️ INCREASED VIGILANCE", "Monitor closely and prepare for potential action"
    elif risk_level == "Medium" or probability >= 0.4:
        return "👀 STANDARD MONITORING", "Maintain regular patrol awareness"
    else:
        return "✅ ROUTINE PATROL", "Normal patrol operations sufficient"

@login_required
def main():
    st.set_page_config(
        page_title="Crime Prediction - CrimeCast", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply some basic styling
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
        }
        .stAlert {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🔮 Crime Probability Predictor")
    
    # Initialize session state for coordinates
    if 'latitude' not in st.session_state:
        st.session_state['latitude'] = 41.8781
    if 'longitude' not in st.session_state:
        st.session_state['longitude'] = -87.6298
    
    if not PREDICTOR_AVAILABLE:
        st.error("❌ Prediction system not available. Please check dependencies.")
        st.info("💡 Make sure all required packages are installed and try again.")
        return
        
    predictor = load_models()
    
    if predictor is None:
        st.error("❌ Models not loaded. Please ensure models are trained and available.")
        st.info("💡 If running locally, make sure your model files are in the 'models' directory.")
        return
    
    st.markdown("Provide the details below to get a real-time prediction of arrest probability.")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location Information")
        
        address = st.text_input(
            "Address or Location", 
            value="1060 W Addison St",
            help="Enter a Chicago address",
            key="address_input"
        )
        
        if st.button("🔍 Get Coordinates", key="geocode_btn", use_container_width=True):
            with st.spinner("Finding location..."):
                lat, lon, error = safe_geocode_address(address)
                if error:
                    st.error(f"❌ {error}")
                    if "timeout" in error.lower():
                        st.info("💡 Geocoding service is slow. You can manually enter coordinates below.")
                else:
                    st.session_state['latitude'] = lat
                    st.session_state['longitude'] = lon
                    st.success(f"✓ Found: {lat:.4f}°N, {lon:.4f}°W")
        
        latitude = st.number_input(
            "Latitude", 
            value=st.session_state.get('latitude', 41.8781), 
            min_value=41.6, 
            max_value=42.1, 
            format="%.6f",
            key="lat_input"
        )
        longitude = st.number_input(
            "Longitude", 
            value=st.session_state.get('longitude', -87.6298), 
            min_value=-87.95, 
            max_value=-87.5, 
            format="%.6f",
            key="lon_input"
        )
        beat = st.number_input("Beat", value=1032, min_value=111, max_value=2535, key="beat_input")
        district = st.number_input("District", value=10, min_value=1, max_value=31, key="district_input")
    
    with col2:
        st.subheader("🏛️ Area Information")
        ward = st.number_input("Ward", value=23, min_value=1, max_value=50, key="ward_input")
        community_area = st.number_input("Community Area", value=32, min_value=1, max_value=77, key="community_input")
        location_type = st.selectbox(
            "Location Type", 
            ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER", 
             "PARKING LOT", "ALLEY", "COMMERCIAL", "SCHOOL"],
            key="location_type_input"
        )
    
    with col3:
        st.subheader("⏰ Time Information")
        year = st.number_input("Year", value=2020, min_value=2001, max_value=2023, key="year_input")
        month = st.slider("Month", 1, 12, 6, key="month_slider")
        hour = st.slider("Hour of Day", 0, 23, 12, key="hour_slider")
        day_of_week = st.selectbox(
            "Day of Week", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            key="day_input"
        )
    
    # Auto-calculate derived features
    st.subheader("🔄 Calculated Features")
    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
        if 5 <= hour < 12:
            time_of_day = "Morning"
        elif 12 <= hour < 17:
            time_of_day = "Afternoon"
        elif 17 <= hour < 21:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"
        st.info(f"**Time of Day**: {time_of_day}")
    
    with col_calc2:
        if month in [12, 1, 2]:
            season = "Winter"
        elif month in [3, 4, 5]:
            season = "Spring"
        elif month in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"
        st.info(f"**Season**: {season}")
    
    day_of_week_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
    
    # Prediction section
    st.markdown("---")
    st.subheader("🎯 Get Prediction")
    
    if st.button("🔮 Predict Arrest Probability", type="primary", use_container_width=True, key="predict_btn"):
        with st.spinner("🤖 Analyzing crime patterns with AI..."):
            try:
                features = {
                    'Latitude': float(latitude),
                    'Longitude': float(longitude),
                    'Beat': int(beat),
                    'District': int(district),
                    'Ward': int(ward),
                    'Community Area': int(community_area),
                    'Hour': int(hour),
                    'DayOfWeek': int(day_of_week_num),
                    'Month': int(month),
                    'Year': int(year),
                    'Location_Description_Clean': str(location_type),
                    'TimeOfDay': str(time_of_day),
                    'Season': str(season)
                }
                
                result = predictor.predict(features)
                
                if 'error' not in result:
                    st.success("🎉 Prediction Completed!")
                    
                    # Get police-friendly metrics
                    reliability_level, reliability_desc = get_prediction_reliability(result['probability'])
                    operational_priority, priority_desc = get_operational_priority(result['probability'], result['risk_level'])
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        prob_display = f"{result['probability']:.1%}"
                        st.metric("Arrest Probability", prob_display)
                    
                    with col2:
                        prediction_text = "🚨 ARREST" if result['prediction'] == 1 else "✅ NO ARREST"
                        st.metric("Prediction", prediction_text)
                    
                    with col3:
                        st.metric("Risk Level", result['risk_level'])
                    
                    with col4:
                        st.metric("Prediction Reliability", reliability_level)
                    
                    # Operational Priority Section
                    st.subheader("🎯 Operational Assessment")
                    priority_col1, priority_col2 = st.columns(2)
                    
                    with priority_col1:
                        st.info(f"**Operational Priority**: {operational_priority}")
                        st.write(f"*{priority_desc}*")
                    
                    with priority_col2:
                        st.info(f"**Prediction Reliability**: {reliability_level}")
                        st.write(f"*{reliability_desc}*")
                    
                    # Visualization
                    st.subheader("📈 Probability Visualization")
                    fig, ax = plt.subplots(figsize=(12, 3))
                    
                    color_map = {'High': '#ff4b4b', 'Medium': '#ffa500', 'Low': '#00cc96'}
                    bar_color = color_map.get(result['risk_level'], '#1f77b4')
                    
                    bars = ax.barh(['Arrest Probability'], [result['probability']], 
                                  color=bar_color, alpha=0.8, height=0.6)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability Scale')
                    ax.set_title(f'Arrest Probability: {result["probability"]:.1%}', fontweight='bold')
                    
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                               f'{width:.1%}', ha='left', va='center', 
                               fontweight='bold', fontsize=16, color=bar_color)
                    
                    ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Low Risk')
                    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Risk')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()  # Prevent memory leaks
                    
                    # KEY FACTORS SECTION
                    st.subheader("🔑 Key Factors Influencing This Prediction")
                    
                    # Get feature importance
                    feature_importance = get_feature_importance(predictor, features)
                    
                    # Display key factors
                    col_f1, col_f2 = st.columns(2)
                    
                    with col_f1:
                        st.markdown("### 📊 Feature Importance")
                        # Create horizontal bar chart for feature importance
                        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                        features_sorted = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                        
                        y_pos = np.arange(len(features_sorted))
                        ax_imp.barh(y_pos, list(features_sorted.values()), color='#667eea', alpha=0.8)
                        ax_imp.set_yticks(y_pos)
                        ax_imp.set_yticklabels(list(features_sorted.keys()))
                        ax_imp.set_xlabel('Importance Score')
                        ax_imp.set_title('Key Factors Influencing Prediction')
                        ax_imp.grid(True, alpha=0.3, axis='x')
                        
                        # Add value labels on bars
                        for i, v in enumerate(features_sorted.values()):
                            ax_imp.text(v + 0.01, i, f'{v:.2f}', va='center', fontweight='bold')
                        
                        st.pyplot(fig_imp)
                        plt.close()  # Prevent memory leaks
                    
                    with col_f2:
                        st.markdown("### 💡 Risk Indicators")
                        
                        # Risk factors analysis
                        risk_factors = []
                        
                        # Analyze time factors
                        if time_of_day == 'Night':
                            risk_factors.append("🌙 **Night Time**: Higher risk period")
                        elif time_of_day == 'Evening':
                            risk_factors.append("🌆 **Evening Hours**: Moderate risk period")
                        
                        # Analyze location factors
                        if location_type in ['ALLEY', 'PARKING LOT']:
                            risk_factors.append("📍 **High-Risk Location**: Areas with limited visibility")
                        elif location_type in ['RESIDENCE', 'SCHOOL']:
                            risk_factors.append("🏠 **Controlled Environment**: Lower risk location")
                        
                        # Analyze day factors
                        if day_of_week in ['Friday', 'Saturday']:
                            risk_factors.append("🎉 **Weekend**: Higher activity periods")
                        
                        # Analyze seasonal factors
                        if season == 'Summer':
                            risk_factors.append("☀️ **Summer Season**: Typically higher crime rates")
                        
                        # Display risk factors
                        if risk_factors:
                            for factor in risk_factors:
                                st.write(f"• {factor}")
                        else:
                            st.info("📊 No significant risk factors identified for this scenario.")
                        
                        # Recommendations based on operational priority
                        st.markdown("### 🛡️ Operational Recommendations")
                        if "IMMEDIATE ACTION" in operational_priority:
                            st.error("""
                            **🚨 IMMEDIATE ACTION REQUIRED:**
                            - Increase patrol presence immediately
                            - Deploy additional units to area
                            - Activate surveillance if available
                            - Prepare for rapid response
                            - Notify command center
                            """)
                        elif "INCREASED VIGILANCE" in operational_priority:
                            st.warning("""
                            **⚠️ INCREASED VIGILANCE:**
                            - Maintain visible patrol presence
                            - Conduct frequent area checks
                            - Monitor for suspicious activity
                            - Document all observations
                            - Stay alert for escalation
                            """)
                        elif "STANDARD MONITORING" in operational_priority:
                            st.info("""
                            **👀 STANDARD MONITORING:**
                            - Continue regular patrol patterns
                            - Maintain situational awareness
                            - Report any unusual activity
                            - Engage with community members
                            """)
                        else:
                            st.success("""
                            **✅ ROUTINE PATROL:**
                            - Normal patrol operations sufficient
                            - Focus on community engagement
                            - Continue preventive measures
                            - Maintain standard protocols
                            """)
                
                else:
                    st.error(f"❌ Prediction error: {result['error']}")
                    
            except Exception as e:
                st.error(f"❌ Unexpected error during prediction: {str(e)}")
                st.info("💡 Please check your input values and try again.")

if __name__ == "__main__":
    main()