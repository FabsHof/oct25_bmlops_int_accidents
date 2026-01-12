"""
Severity Legend Component
Displays a visual legend explaining the severity levels.
"""

import streamlit as st
import sys
from pathlib import Path

# Add streamlit directory to path for imports
STREAMLIT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STREAMLIT_DIR))

from config import SEVERITY_CLASSES


def render_severity_legend():
    """Render a styled severity legend with icons and descriptions."""
    st.markdown("### 📊 Severity Levels")
    
    cols = st.columns(4)
    for idx, (level, info) in enumerate(SEVERITY_CLASSES.items()):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="
                    background-color: {info['color']}20;
                    border-left: 4px solid {info['color']};
                    padding: 1rem;
                    border-radius: 0.5rem;
                    height: 140px;
                ">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{info['icon']}</div>
                    <div style="font-weight: bold; color: {info['color']};">Level {level}</div>
                    <div style="font-weight: 600;">{info['label']}</div>
                    <div style="font-size: 0.85rem; color: #666;">{info['description']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
