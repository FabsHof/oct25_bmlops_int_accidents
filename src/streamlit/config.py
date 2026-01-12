"""
Configuration module for the Streamlit application.
Centralizes environment variables and constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# API Configuration
# ==============================================================================
API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = os.environ.get("API_PORT", "8000")
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# ==============================================================================
# Severity Classes
# ==============================================================================
SEVERITY_CLASSES = {
    1: {"label": "Unscathed", "color": "#28a745", "icon": "✅", "description": "No injury"},
    2: {"label": "Light injury", "color": "#ffc107", "icon": "⚠️", "description": "Minor injuries, no hospitalization required"},
    3: {"label": "Hospitalized wounded", "color": "#fd7e14", "icon": "🏥", "description": "Serious injuries requiring hospitalization"},
    4: {"label": "Killed", "color": "#dc3545", "icon": "💀", "description": "Fatal outcome"},
}

# ==============================================================================
# Feature Options for Prediction Form
# ==============================================================================
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

USER_CATEGORIES = {
    "Driver": 1,
    "Passenger": 2,
    "Pedestrian": 3,
    "Pedestrian on rollerblades or scooter": 4
}

GENDERS = {"Male": 1, "Female": 2}

TRIP_PURPOSES = {
    "Unknown": 0,
    "Work commuting": 1,
    "School commuting": 2,
    "Shopping": 3,
    "Professional use": 4,
    "Leisure": 5,
    "Other": 9
}

SAFETY_EQUIPMENT = {
    "Unknown": 0,
    "Belt": 1,
    "Helmet": 2,
    "Children's device": 3,
    "Reflective Equipment": 4,
    "Other": 9
}

LUMINOSITY = {
    "Full day": 1,
    "Twilight or dawn": 2,
    "Night without public lighting": 3,
    "Night with public lighting not lit": 4,
    "Night with public lighting on": 5
}

WEATHER_CONDITIONS = {
    "Normal": 1,
    "Light rain": 2,
    "Heavy rain": 3,
    "Snow / hail": 4,
    "Fog / smoke": 5,
    "Strong wind / storm": 6,
    "Dazzling weather": 7,
    "Cloudy weather": 8,
    "Other": 9
}

ROAD_TYPES = {
    "Highway": 1,
    "National road": 2,
    "Departmental road": 3,
    "Communal way": 4,
    "Off public network": 5,
    "Parking lot open to public traffic": 6,
    "Other": 9
}

ROAD_SURFACES = {
    "Normal": 1,
    "Wet": 2,
    "Puddles": 3,
    "Flooded": 4,
    "Snow": 5,
    "Mud": 6,
    "Icy": 7,
    "Grease / Oil": 8,
    "Other": 9
}

HOLIDAYS = {"No": 0, "Yes": 1}

# ==============================================================================
# UI Configuration
# ==============================================================================
PAGE_ICON = "🚗"
PAGE_TITLE = "Accident Severity Prediction"
