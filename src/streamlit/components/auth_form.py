"""
Authentication Form Component
Handles OAuth2 authentication with the API.
"""

import streamlit as st
import requests
import sys
from pathlib import Path

# Add streamlit directory to path for imports
STREAMLIT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STREAMLIT_DIR))

from config import API_BASE_URL


def render_auth_form():
    """
    Render an authentication form with improved UX.
    Returns True if authenticated, False otherwise.
    """
    # Check if already authenticated
    if st.session_state.get("authenticated", False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ Authenticated successfully!")
        with col2:
            if st.button("🚪 Logout", type="secondary"):
                st.session_state.authenticated = False
                st.session_state.auth_headers = {}
                st.session_state.username = None
                st.rerun()
        return True
    
    # Authentication form
    st.markdown("### 🔐 Authentication Required")
    st.info("Please enter your credentials to access the prediction service.")
    
    with st.form("auth_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Contact your administrator for credentials"
            )
        with col2:
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )
        
        submitted = st.form_submit_button("🔑 Sign In", type="primary", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
                return False
            
            with st.spinner("Authenticating..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/token",
                        data={"username": username, "password": password},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        token_data = response.json()
                        st.session_state.auth_headers = {
                            "Authorization": f"Bearer {token_data['access_token']}"
                        }
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("Authentication successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
                        
                except requests.ConnectionError:
                    st.error("🔌 Could not connect to the API server. Please try again later.")
                except requests.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {e}")
    
    return False
