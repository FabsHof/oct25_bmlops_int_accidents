"""
Prediction Result Component
Displays model prediction results with enhanced visualization.
"""

import streamlit as st
import sys
from pathlib import Path

# Add streamlit directory to path for imports
STREAMLIT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STREAMLIT_DIR))

from config import SEVERITY_CLASSES


def render_prediction_result(prediction_response):
    """
    Render prediction results with visual enhancements.
    
    Args:
        prediction_response: PredictionResponse object with prediction data
    """
    pred_level = prediction_response.prediction
    severity_info = SEVERITY_CLASSES.get(pred_level, SEVERITY_CLASSES[1])
    
    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")
    
    # Main result card
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {severity_info['color']}20, {severity_info['color']}40);
            border: 2px solid {severity_info['color']};
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
        ">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">{severity_info['icon']}</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {severity_info['color']};">
                {prediction_response.prediction_label}
            </div>
            <div style="font-size: 1rem; color: #666; margin-top: 0.5rem;">
                Confidence: <strong>{prediction_response.confidence:.1%}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Probability breakdown
    st.markdown("### 📊 Probability Distribution")
    
    if prediction_response.probabilities:
        cols = st.columns(4)
        sorted_probs = sorted(
            prediction_response.probabilities.items(),
            key=lambda x: list(SEVERITY_CLASSES.values()).index(
                next((v for v in SEVERITY_CLASSES.values() if v["label"] == x[0]), SEVERITY_CLASSES[1])
            ) if any(v["label"] == x[0] for v in SEVERITY_CLASSES.values()) else 0
        )
        
        for idx, (label, prob) in enumerate(sorted_probs):
            severity = next(
                (v for v in SEVERITY_CLASSES.values() if v["label"] == label),
                {"color": "#888", "icon": "•"}
            )
            with cols[idx % 4]:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f8f9fa;
                        border-radius: 0.5rem;
                        padding: 0.75rem;
                        text-align: center;
                        margin-bottom: 0.5rem;
                    ">
                        <div style="font-size: 1.5rem;">{severity.get('icon', '•')}</div>
                        <div style="font-size: 0.85rem; color: #666;">{label}</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: {severity.get('color', '#888')};">
                            {prob:.1%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Progress bars for probabilities
        st.markdown("#### Detailed Breakdown")
        for label, prob in sorted_probs:
            severity = next(
                (v for v in SEVERITY_CLASSES.values() if v["label"] == label),
                {"color": "#888"}
            )
            st.progress(prob, text=f"{label}: {prob:.1%}")
    
    # Model info expander
    with st.expander("ℹ️ Model Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Version", prediction_response.model_version or "N/A")
        with col2:
            st.metric("Confidence Score", f"{prediction_response.confidence:.2%}")
