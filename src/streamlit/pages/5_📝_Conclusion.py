"""
Conclusion Page

Summary of the project and future outlook.
"""

import streamlit as st

st.set_page_config(
    page_title="Conclusion - Accident Severity",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Conclusion & Outlook")
st.markdown("Summary of achievements and future directions.")

st.markdown("---")

# Achievements section
st.markdown("## 🏆 What We Achieved")

achievements = [
    {
        "icon": "🔧",
        "title": "Reproducible Development Environment",
        "description": "Set up a fully reproducible environment with UV and GitHub for team collaboration"
    },
    {
        "icon": "🏗️",
        "title": "Complete MLOps Architecture",
        "description": "Designed and implemented end-to-end pipeline from data ingestion to model prediction"
    },
    {
        "icon": "🧪",
        "title": "Quality Assurance",
        "description": "Implemented unit tests and monitoring to ensure correct execution"
    },
    {
        "icon": "🐳",
        "title": "Containerized Deployment",
        "description": "Entire architecture deployable with a single command via Docker Compose"
    },
    {
        "icon": "🤖",
        "title": "Automated ML Pipeline",
        "description": "Airflow DAGs orchestrate data processing, training, and model promotion"
    },
    {
        "icon": "🔐",
        "title": "Secure API Access",
        "description": "OAuth2 authentication protects prediction endpoints"
    }
]

cols = st.columns(3)
for idx, achievement in enumerate(achievements):
    with cols[idx % 3]:
        st.markdown(
            f"""
            <div style="
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 0.75rem;
                padding: 1.5rem;
                margin-bottom: 1rem;
                height: 180px;
            ">
                <div style="font-size: 2rem; text-align: center; margin-bottom: 0.5rem;">
                    {achievement['icon']}
                </div>
                <div style="font-weight: bold; color: #1E3A5F; text-align: center; margin-bottom: 0.5rem;">
                    {achievement['title']}
                </div>
                <div style="font-size: 0.85rem; color: #666; text-align: center;">
                    {achievement['description']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# Lessons Learned
st.markdown("## 📚 Lessons Learned")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### Technical Insights
        
        - **Docker Optimization**: Resource management is crucial for complex multi-service setups
        - **MLflow + MinIO**: S3-compatible storage adds flexibility but requires careful configuration
        - **Airflow 3.x**: New features require adaptation but offer improved functionality
        - **Streamlit Sessions**: State management requires explicit handling for auth tokens
        """
    )

with col2:
    st.markdown(
        """
        ### Process Insights
        
        - **Incremental Development**: Building services one at a time aids debugging
        - **Documentation**: Inline comments and READMEs save time later
        - **Testing Early**: Unit tests catch issues before they compound
        - **Team Coordination**: Clear task ownership prevents conflicts
        """
    )

st.markdown("---")

# Future Outlook
st.markdown("## 🔮 Future Improvements")

future_items = [
    {
        "category": "Scalability",
        "items": [
            "Kubernetes deployment for horizontal scaling",
            "Load balancing for API endpoints",
            "Database read replicas for query performance"
        ],
        "priority": "High"
    },
    {
        "category": "CI/CD",
        "items": [
            "GitHub Actions for automated testing",
            "Automatic deployment on merge to main",
            "Container image versioning and registry"
        ],
        "priority": "High"
    },
    {
        "category": "Model Improvements",
        "items": [
            "Hyperparameter tuning",
            "Ensemble methods for better accuracy",
            "SHAP values for explainability"
        ],
        "priority": "Medium"
    },
    {
        "category": "Monitoring",
        "items": [
            "More comprehensive Grafana dashboards",
            "Automated alerting on model degradation",
            "A/B testing infrastructure"
        ],
        "priority": "Medium"
    }
]

for item in future_items:
    priority_color = "#dc3545" if item["priority"] == "High" else "#ffc107"
    
    with st.expander(f"{item['category']} (Priority: {item['priority']})", expanded=True):
        for task in item["items"]:
            st.markdown(f"- {task}")

st.markdown("---")

# Thank you section
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1E3A5F, #3d5a80);
        color: white;
        border-radius: 1rem;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
    ">
        <h1 style="color: white; margin: 0;">🙏 Thank You!</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.9;">
            We appreciate your interest in our Accident Severity Prediction project.
        </p>
        <p style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.8;">
            For questions or feedback, please reach out through GitHub Issues.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Final note
st.info(
    "💡 This project demonstrates a complete MLOps workflow for machine learning "
    "in production. While focused on accident severity prediction, the architecture "
    "patterns are applicable to many other ML use cases."
)
