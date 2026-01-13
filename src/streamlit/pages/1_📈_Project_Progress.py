"""
Project Progress Page

Shows the development history and milestones of the project.
"""

import streamlit as st

st.set_page_config(
    page_title="Project Progress - Accident Severity",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Project Progress")
st.markdown("A timeline of our project development and key milestones.")

st.markdown("---")

# Timeline section
st.markdown("## 🗓️ Development Timeline")

milestones = [
    {
        "phase": "Phase 1: Planning & Setup",
        "icon": "📋",
        "items": [
            "Project kickoff and requirements gathering",
            "Team organization using GitHub Projects",
            "Development environment setup with UV",
            "Initial repository structure"
        ],
        "status": "completed"
    },
    {
        "phase": "Phase 2: Data & Infrastructure",
        "icon": "🗄️",
        "items": [
            "PostgreSQL database setup",
            "Data ingestion pipeline from Kaggle",
            "Data preprocessing and cleaning",
            "Docker containerization of services"
        ],
        "status": "completed"
    },
    {
        "phase": "Phase 3: ML Pipeline",
        "icon": "🤖",
        "items": [
            "Feature engineering",
            "Model training and evaluation",
            "MLflow integration for tracking",
            "MinIO for artifact storage"
        ],
        "status": "completed"
    },
    {
        "phase": "Phase 4: API & Frontend",
        "icon": "🌐",
        "items": [
            "FastAPI prediction endpoint",
            "OAuth2 authentication",
            "Streamlit user interface",
            "Integration testing"
        ],
        "status": "completed"
    },
    {
        "phase": "Phase 5: Monitoring & Automation",
        "icon": "📊",
        "items": [
            "Prometheus metrics collection",
            "Grafana dashboards",
            "Evidently drift detection",
            "Airflow DAG orchestration"
        ],
        "status": "completed"
    },
    {
        "phase": "Phase 6: Documentation & Polish",
        "icon": "📝",
        "items": [
            "README documentation",
            "Code cleanup and comments",
            "Unit test coverage",
            "Final presentation preparation"
        ],
        "status": "in-progress"
    }
]

for milestone in milestones:
    status_color = "#28a745" if milestone["status"] == "completed" else "#ffc107"
    status_icon = "✅" if milestone["status"] == "completed" else "🔄"
    
    st.markdown(
        f"""
        <div style="
            background-color: #f8f9fa;
            border-left: 4px solid {status_color};
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{milestone['icon']}</span>
                <span style="font-size: 1.2rem; font-weight: bold; color: #1E3A5F;">
                    {milestone['phase']}
                </span>
                <span style="margin-left: auto; font-size: 1.2rem;">{status_icon}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.expander(f"View details for {milestone['phase']}", expanded=False):
        for item in milestone["items"]:
            st.markdown(f"- {item}")

st.markdown("---")

# Key Achievements
st.markdown("## 🏆 Key Achievements")

cols = st.columns(4)

achievements = [
    {"number": "6+", "label": "Services", "icon": "🐳"},
    {"number": "3", "label": "Airflow DAGs", "icon": "🔄"},
    {"number": "16", "label": "Features", "icon": "📊"},
    {"number": "4", "label": "Severity Classes", "icon": "🎯"},
]

for idx, achievement in enumerate(achievements):
    with cols[idx]:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1E3A5F, #3d5a80);
                color: white;
                border-radius: 0.75rem;
                padding: 1.5rem;
                text-align: center;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.25rem;">{achievement['icon']}</div>
                <div style="font-size: 2rem; font-weight: bold;">{achievement['number']}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">{achievement['label']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# Challenges Section
st.markdown("## 💪 Challenges Overcome")

challenges = [
    {
        "challenge": "Complex Multi-Service Orchestration",
        "solution": "Docker Compose with careful dependency management and health checks"
    },
    {
        "challenge": "MLflow + MinIO Integration",
        "solution": "Custom S3-compatible artifact storage configuration"
    },
    {
        "challenge": "Session State Management in Streamlit",
        "solution": "Careful state handling for OAuth2 token persistence"
    },
    {
        "challenge": "Resource Constraints on Development Machines",
        "solution": "Optimized Docker builds and selective service startup"
    }
]

for ch in challenges:
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**🔴 Challenge:**")
            st.markdown(ch["challenge"])
        with col2:
            st.markdown(f"**🟢 Solution:**")
            st.markdown(ch["solution"])
        st.markdown("---")

# Team section
st.markdown("## 👥 Team Contributions")
st.info("This project was developed as a collaborative effort with contributions across all MLOps components.")
