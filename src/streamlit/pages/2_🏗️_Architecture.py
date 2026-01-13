"""
Architecture Overview Page

Displays the MLOps architecture diagram using Mermaid.
"""

import streamlit as st
import sys
from pathlib import Path

# Add streamlit directory to path for imports
STREAMLIT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STREAMLIT_DIR))

# Add project root for src imports
PROJECT_ROOT = STREAMLIT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.schemas import render_mermaid, overview_code

st.set_page_config(
    page_title="Architecture - Accident Severity",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Architecture Overview")
st.markdown("Explore our MLOps infrastructure from data ingestion to model prediction.")

st.markdown("---")

# Architecture intro
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ## System Components
        
        Our architecture follows MLOps best practices with the following key components:
        
        - **Data Layer**: PostgreSQL database for structured storage
        - **ML Pipeline**: Automated training with Airflow orchestration  
        - **Model Registry**: MLflow for experiment tracking and versioning
        - **Artifact Storage**: MinIO (S3-compatible) for model artifacts
        - **API Layer**: FastAPI with OAuth2 authentication
        - **Monitoring**: Prometheus + Grafana for metrics and alerting
        - **Frontend**: Streamlit for user interaction
        """
    )

with col2:
    st.markdown(
        """
        <div style="
            background-color: #e8f4f8;
            border: 2px solid #1E3A5F;
            border-radius: 0.75rem;
            padding: 1.5rem;
        ">
            <h4 style="color: #1E3A5F; margin-top: 0;">🔧 Technologies</h4>
            <ul style="margin-bottom: 0;">
                <li>Python 3.11</li>
                <li>Docker Compose</li>
                <li>PostgreSQL</li>
                <li>FastAPI</li>
                <li>MLflow 3.x</li>
                <li>Airflow 3.x</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Architecture diagram
st.markdown("## 📊 Architecture Diagram")
st.markdown("Interactive flow diagram showing data and model pipelines:")

render_mermaid(overview_code)

st.markdown("---")

# Component legend
st.markdown("## 🎨 Component Legend")

legend_items = [
    {"color": "#ff6f00", "name": "User Application", "description": "Frontend interface for predictions"},
    {"color": "#ffd54f", "name": "Airflow/CRON", "description": "Workflow orchestration and scheduling"},
    {"color": "#00c853", "name": "API", "description": "FastAPI prediction and training endpoints"},
    {"color": "#2962ff", "name": "Database", "description": "PostgreSQL data storage"},
    {"color": "#00b0ff", "name": "MLflow", "description": "Model tracking and registry"},
]

cols = st.columns(len(legend_items))
for idx, item in enumerate(legend_items):
    with cols[idx]:
        st.markdown(
            f"""
            <div style="
                background-color: {item['color']}20;
                border: 2px solid {item['color']};
                border-radius: 0.5rem;
                padding: 1rem;
                text-align: center;
                height: 120px;
            ">
                <div style="
                    width: 20px;
                    height: 20px;
                    background-color: {item['color']};
                    border-radius: 50%;
                    margin: 0 auto 0.5rem;
                "></div>
                <div style="font-weight: bold; color: #333;">{item['name']}</div>
                <div style="font-size: 0.8rem; color: #666;">{item['description']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# Data flow explanation
st.markdown("## 🔄 Data Flow")

with st.expander("📥 Data Ingestion Flow", expanded=True):
    st.markdown(
        """
        1. **Raw Data** → Downloaded from Kaggle dataset
        2. **Preprocessing** → Cleaned and transformed
        3. **Storage** → Persisted in PostgreSQL database
        4. **Versioning** → Data versions tracked for reproducibility
        """
    )

with st.expander("🤖 Model Training Flow"):
    st.markdown(
        """
        1. **Data Fetch** → Retrieve preprocessed data from database
        2. **Feature Engineering** → Prepare features for training
        3. **Model Training** → Train Random Forest classifier
        4. **Evaluation** → Calculate metrics (F1, accuracy, etc.)
        5. **Registration** → Log model and metrics to MLflow
        6. **Promotion** → Best model tagged as 'champion'
        """
    )

with st.expander("🎯 Prediction Flow"):
    st.markdown(
        """
        1. **User Input** → Features submitted via Streamlit
        2. **Authentication** → OAuth2 token validation
        3. **API Request** → FastAPI receives prediction request
        4. **Model Load** → Champion model loaded from MLflow
        5. **Inference** → Model generates severity prediction
        6. **Response** → Prediction with probabilities returned to user
        """
    )
