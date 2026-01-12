"""
Prediction Page - Accident Severity Prediction

Allows users to input accident parameters and receive severity predictions.
"""

import datetime as dt
import streamlit as st
import requests
import sys
from pathlib import Path

# Add streamlit directory to path for imports
STREAMLIT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STREAMLIT_DIR))

# Add project root for src imports
PROJECT_ROOT = STREAMLIT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    API_BASE_URL,
    SEVERITY_CLASSES,
    MONTHS,
    USER_CATEGORIES,
    GENDERS,
    TRIP_PURPOSES,
    SAFETY_EQUIPMENT,
    LUMINOSITY,
    WEATHER_CONDITIONS,
    ROAD_TYPES,
    ROAD_SURFACES,
    HOLIDAYS,
)
from components.severity_legend import render_severity_legend
from components.auth_form import render_auth_form
from components.prediction_result import render_prediction_result

from src.utils.ml_utils import PredictionRequest, PredictionResponse

st.set_page_config(
    page_title="Prediction - Accident Severity",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Accident Severity Prediction")
st.markdown("Input accident parameters to predict the severity level.")

st.markdown("---")

# Severity Legend
render_severity_legend()

st.markdown("---")

# Authentication Section
st.markdown("## Step 1: Authentication")
is_authenticated = render_auth_form()

if not is_authenticated:
    st.warning("⚠️ Please authenticate to access the prediction service.")
    st.stop()

st.markdown("---")

# Prediction Form
st.markdown("## Step 2: Enter Accident Parameters")

# Initialize default prediction request
default_features = PredictionRequest(
    year=2012,
    month=6,
    hour=14,
    minute=30,
    user_category=1,
    sex=1,
    year_of_birth=1990,
    trip_purpose=1,
    security=1,
    luminosity=1,
    weather=1,
    type_of_road=1,
    road_surface=1,
    latitude=48.8566,
    longitude=2.3522,
    holiday=0
)

# Create tabs for organized input
tab_time, tab_person, tab_conditions, tab_location = st.tabs([
    "🕐 Time & Date",
    "👤 Person Details", 
    "🌤️ Conditions",
    "📍 Location"
])

with tab_time:
    st.markdown("### When did the accident occur?")
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.slider(
            "Year of the accident",
            min_value=2006,
            max_value=2016,
            value=2012,
            help="Select the year when the accident occurred"
        )
        
        month_name = st.select_slider(
            "Month of the accident",
            options=list(MONTHS.keys()),
            value="June"
        )
        month = MONTHS[month_name]
    
    with col2:
        time_input = st.time_input(
            "Time of the accident",
            value=dt.time(14, 0),
            help="24-hour format"
        )
        hour = time_input.hour
        minute = time_input.minute
        
        holiday_key = st.radio(
            "During holidays?",
            options=list(HOLIDAYS.keys()),
            horizontal=True
        )
        holiday = HOLIDAYS[holiday_key]

with tab_person:
    st.markdown("### Who was involved?")
    col1, col2 = st.columns(2)
    
    with col1:
        user_category_key = st.selectbox(
            "User Category",
            options=list(USER_CATEGORIES.keys()),
            help="Role of the person in the accident"
        )
        user_category = USER_CATEGORIES[user_category_key]
        
        sex_key = st.selectbox(
            "Gender",
            options=list(GENDERS.keys())
        )
        sex = GENDERS[sex_key]
    
    with col2:
        year_of_birth = st.slider(
            "Year of birth",
            min_value=1920,
            max_value=2010,
            value=1985,
            help="Birth year of the person involved"
        )
        
        trip_purpose_key = st.selectbox(
            "Trip Purpose",
            options=list(TRIP_PURPOSES.keys())
        )
        trip_purpose = TRIP_PURPOSES[trip_purpose_key]
    
    security_key = st.selectbox(
        "Safety Equipment",
        options=list(SAFETY_EQUIPMENT.keys()),
        help="Primary safety equipment used"
    )
    security = SAFETY_EQUIPMENT[security_key]

with tab_conditions:
    st.markdown("### What were the conditions?")
    col1, col2 = st.columns(2)
    
    with col1:
        luminosity_key = st.selectbox(
            "Lighting Conditions",
            options=list(LUMINOSITY.keys())
        )
        luminosity = LUMINOSITY[luminosity_key]
        
        weather_key = st.selectbox(
            "Weather Conditions",
            options=list(WEATHER_CONDITIONS.keys())
        )
        weather = WEATHER_CONDITIONS[weather_key]
    
    with col2:
        road_type_key = st.selectbox(
            "Type of Road",
            options=list(ROAD_TYPES.keys())
        )
        type_of_road = ROAD_TYPES[road_type_key]
        
        road_surface_key = st.selectbox(
            "Road Surface Condition",
            options=list(ROAD_SURFACES.keys())
        )
        road_surface = ROAD_SURFACES[road_surface_key]

with tab_location:
    st.markdown("### Where did it happen?")
    st.info("💡 Default coordinates are set to central Paris.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        latitude = st.number_input(
            "Latitude",
            min_value=41.0,
            max_value=51.5,
            value=48.8566,
            format="%.4f",
            help="Latitude in decimal degrees (France: ~41-51)"
        )
    
    with col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-5.5,
            max_value=10.0,
            value=2.3522,
            format="%.4f",
            help="Longitude in decimal degrees (France: ~-5 to 10)"
        )
    
    # Simple map visualization
    st.map(data={"lat": [latitude], "lon": [longitude]}, zoom=5)

st.markdown("---")

# Build prediction request
features = PredictionRequest(
    year=year,
    month=month,
    hour=hour,
    minute=minute,
    user_category=user_category,
    sex=sex,
    year_of_birth=year_of_birth,
    trip_purpose=trip_purpose,
    security=security,
    luminosity=luminosity,
    weather=weather,
    type_of_road=type_of_road,
    road_surface=road_surface,
    latitude=latitude,
    longitude=longitude,
    holiday=holiday
)

# Prediction Section
st.markdown("## Step 3: Get Prediction")

col1, col2 = st.columns([2, 1])

with col1:
    predict_clicked = st.button(
        "🚀 Get Severity Prediction",
        type="primary",
        use_container_width=True,
        help="Submit the parameters to get a prediction"
    )

with col2:
    model_info_clicked = st.button(
        "ℹ️ Model Information",
        use_container_width=True,
        help="View current model details"
    )

if predict_clicked:
    with st.spinner("🔄 Analyzing accident parameters..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=features.model_dump(),
                headers=st.session_state.get("auth_headers", {}),
                timeout=30
            )
            
            if response.status_code == 200:
                prediction_response = PredictionResponse(**response.json())
                render_prediction_result(prediction_response)
            elif response.status_code == 401:
                st.error("🔒 Authentication expired. Please log in again.")
                st.session_state.authenticated = False
                st.rerun()
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.ConnectionError:
            st.error("🔌 Could not connect to the API server. Please ensure the service is running.")
        except requests.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

if model_info_clicked:
    with st.spinner("Loading model information..."):
        try:
            response = requests.get(
                f"{API_BASE_URL}/model/info",
                headers=st.session_state.get("auth_headers", {}),
                timeout=10
            )
            
            if response.status_code == 200:
                model_info = response.json()
                
                st.markdown("### 📋 Current Model Information")
                
                if isinstance(model_info, dict):
                    cols = st.columns(3)
                    for idx, (key, value) in enumerate(model_info.items()):
                        with cols[idx % 3]:
                            st.metric(key.replace("_", " ").title(), str(value))
                else:
                    st.json(model_info)
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.ConnectionError:
            st.error("🔌 Could not connect to the API server.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

# Feature summary expander
with st.expander("📋 Current Parameter Summary"):
    st.json(features.model_dump())
