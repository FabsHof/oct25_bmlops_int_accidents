"""
MLOps Components Page

Detailed descriptions of all tools and technologies used in the project.
"""

import streamlit as st
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Components - Accident Severity",
    page_icon="🧩",
    layout="wide"
)

# Get the streamlit directory path
STREAMLIT_DIR = Path(__file__).parent.parent
IMAGES_DIR = STREAMLIT_DIR / "assets" / "images"


def load_image(filename: str, fallback_dir: Path = STREAMLIT_DIR):
    """Try to load image from assets/images or fallback to streamlit dir."""
    image_path = IMAGES_DIR / filename
    if image_path.exists():
        return Image.open(image_path)
    
    fallback_path = fallback_dir / filename
    if fallback_path.exists():
        return Image.open(fallback_path)
    
    return None


def render_component_card(
    title: str,
    what_needed: list,
    selected_tools: dict,
    logo_file: str,
    advantages: list,
    disadvantages: list,
    comments: list = None,
    links: list = None,
    screenshot_file: str = None
):
    """Render a standardized component description card."""
    
    st.markdown(f"### {title}")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 Requirements")
        for item in what_needed:
            st.markdown(f"- {item}")
        
        st.markdown("#### 🛠️ Tool Selection")
        for tool, selected in selected_tools.items():
            if selected:
                st.markdown(f"✅ **{tool}**")
            else:
                st.markdown(f"⬜ {tool}")
    
    with col2:
        logo = load_image(logo_file)
        if logo:
            st.image(logo, width=200)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Advantages")
        for item in advantages:
            st.markdown(f"- {item}")
    
    with col2:
        st.markdown("#### ⚠️ Challenges")
        for item in disadvantages:
            st.markdown(f"- {item}")
    
    if comments:
        st.markdown("#### 💡 Notes")
        for item in comments:
            st.markdown(f"- {item}")
    
    if links:
        st.markdown("#### 🔗 Resources")
        for link in links:
            st.markdown(f"[{link['text']}]({link['url']})")
    
    if screenshot_file:
        with st.expander("📸 Screenshot"):
            screenshot = load_image(screenshot_file)
            if screenshot:
                st.image(screenshot, use_container_width=True)


st.title("🧩 MLOps Components")
st.markdown("Detailed overview of the tools and technologies powering our solution.")

st.markdown("---")

# Component categories
components = {
    "overview": "📋 Components Overview",
    "environment": "💻 Development Environment",
    "data": "🗄️ Data Handling",
    "model": "🤖 ML Model",
    "api": "🌐 Prediction API",
    "mlflow": "📊 Tracking & Versioning",
    "docker": "🐳 Containerization",
    "testing": "🧪 Unit Testing",
    "monitoring": "📈 Drift & Monitoring",
    "automation": "⚙️ Automation",
    "frontend": "🖥️ User Frontend"
}

selected_component = st.selectbox(
    "Select a component to explore:",
    options=list(components.keys()),
    format_func=lambda x: components[x]
)

st.markdown("---")

if selected_component == "overview":
    st.markdown("## 📋 Components Overview")
    
    st.markdown("Our MLOps architecture comprises the following key components:")
    
    cols = st.columns(3)
    component_list = list(components.items())[1:]
    
    for idx, (key, name) in enumerate(component_list):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div style="
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-bottom: 0.5rem;
                    text-align: center;
                ">
                    <div style="font-size: 1.5rem;">{name.split()[0]}</div>
                    <div style="font-size: 0.9rem; color: #1E3A5F;">{' '.join(name.split()[1:])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

elif selected_component == "environment":
    render_component_card(
        title="Development Environment",
        what_needed=[
            "Split project into manageable subtasks",
            "Coordinate team work with realistic timeline",
            "Track development in a centralized manner",
            "Ensure reproducible environment across team"
        ],
        selected_tools={
            "GitHub + GitHub Projects": True,
            "GitLab": False,
            "Astral UV": True,
            "Poetry": False,
            "Conda": False
        },
        logo_file="Logo_github.png",
        advantages=[
            "Git-based collaboration with integrated project management",
            "UV: Extremely fast package management (10-100x faster than pip)",
            "Exact reproducibility with uv.lock file",
            "Cross-platform compatibility"
        ],
        disadvantages=[
            "Initial learning curve for UV syntax",
            "GitHub Projects has limited advanced reporting"
        ],
        comments=[
            "UV reduced environment setup time from minutes to seconds"
        ],
        links=[
            {"text": "UV Documentation", "url": "https://docs.astral.sh/uv/"}
        ],
        screenshot_file="SS_github_project.png"
    )

elif selected_component == "data":
    render_component_card(
        title="Data Handling",
        what_needed=[
            "Store accident data reliably",
            "Handle structured data with relationships",
            "Support SQL queries for data retrieval",
            "Enable data versioning"
        ],
        selected_tools={
            "PostgreSQL": True,
            "SQLite": False,
            "NoSQL (MongoDB)": False
        },
        logo_file="Logo_tool_X.png",
        advantages=[
            "Robust and reliable RDBMS",
            "Excellent performance for structured data",
            "Wide ecosystem and tooling support",
            "Strong consistency and ACID compliance"
        ],
        disadvantages=[
            "Requires more setup than SQLite",
            "Container persistence configuration needed"
        ],
        comments=[
            "Data downloaded from Kaggle and preprocessed before storage",
            "Separate databases for Airflow, MLflow, and accident data"
        ]
    )

elif selected_component == "model":
    render_component_card(
        title="ML Model",
        what_needed=[
            "Classify accident severity into 4 classes",
            "Handle imbalanced dataset",
            "Provide probability estimates",
            "Enable model interpretability"
        ],
        selected_tools={
            "Random Forest Classifier": True,
            "XGBoost": False,
            "Neural Network": False,
            "Logistic Regression": False
        },
        logo_file="Logo_tool_X.png",
        advantages=[
            "Handles categorical and numerical features well",
            "Built-in feature importance",
            "Robust to overfitting",
            "Provides probability estimates"
        ],
        disadvantages=[
            "Less interpretable than linear models",
            "Can be slow with very large datasets"
        ],
        comments=[
            "Weighted F1-score used for evaluation due to class imbalance",
            "Champion model automatically selected based on metrics"
        ]
    )

elif selected_component == "api":
    render_component_card(
        title="Prediction API",
        what_needed=[
            "Expose prediction endpoint",
            "Secure access with authentication",
            "Provide clear API documentation",
            "Handle errors gracefully"
        ],
        selected_tools={
            "FastAPI": True,
            "Flask": False,
            "Django REST": False
        },
        logo_file="Logo_tool_X.png",
        advantages=[
            "Automatic OpenAPI documentation",
            "High performance with async support",
            "Built-in data validation with Pydantic",
            "Easy OAuth2 integration"
        ],
        disadvantages=[
            "Requires understanding of async patterns",
            "Smaller community than Django"
        ],
        links=[
            {"text": "API Documentation (local)", "url": "http://localhost:8000/docs"}
        ]
    )

elif selected_component == "mlflow":
    render_component_card(
        title="Tracking & Versioning",
        what_needed=[
            "Track experiments and metrics",
            "Version models",
            "Store model artifacts",
            "Enable model comparison"
        ],
        selected_tools={
            "MLflow": True,
            "Weights & Biases": False,
            "Neptune": False
        },
        logo_file="Logo_tool_X.png",
        advantages=[
            "Open-source with no vendor lock-in",
            "Model registry with aliases",
            "S3-compatible artifact storage (MinIO)",
            "Built-in model serving capabilities"
        ],
        disadvantages=[
            "Complex initial setup with MinIO",
            "UI can be slow with many experiments"
        ],
        links=[
            {"text": "MLflow UI (default: mlflow/mlflow)", "url": "http://localhost:5001"},
            {"text": "MinIO Console (default: minio_user/minio_password)", "url": "http://localhost:9001"}
        ]
    )

elif selected_component == "docker":
    render_component_card(
        title="Containerization",
        what_needed=[
            "Package services into portable containers",
            "Orchestrate multiple containers",
            "Ensure consistent environments",
            "Simplify deployment"
        ],
        selected_tools={
            "Docker + Docker Compose": True,
            "Podman": False,
            "Kubernetes": False
        },
        logo_file="Logo_docker.jpg",
        advantages=[
            "Widespread adoption and ecosystem",
            "Precise networking and volume management",
            "Single command deployment with Compose",
            "Reproducible environments"
        ],
        disadvantages=[
            "Resource intensive (RAM/disk)",
            "Debugging can be challenging",
            "Build caching complexity"
        ],
        comments=[
            "Optimized builds to reduce resource usage",
            "Health checks ensure proper startup order"
        ],
        screenshot_file="SS_docker.png"
    )

elif selected_component == "testing":
    render_component_card(
        title="Unit Testing",
        what_needed=[
            "Validate service functionality",
            "Catch regressions early",
            "Ensure code quality",
            "Enable confident refactoring"
        ],
        selected_tools={
            "Pytest": True,
            "unittest": False,
            "GitHub Actions": False
        },
        logo_file="Logo_pytest.png",
        advantages=[
            "Clean and readable test syntax",
            "Powerful fixtures system",
            "Good plugin ecosystem",
            "Detailed failure output"
        ],
        disadvantages=[
            "Manual execution (no CI/CD yet)",
            "Mocking complexity for integrations"
        ],
        comments=[
            "Tests organized by module (api, data, models, monitoring)",
            "Future: GitHub Actions for automated testing"
        ],
        screenshot_file="SS_testing.png"
    )

elif selected_component == "monitoring":
    render_component_card(
        title="Drift & Monitoring",
        what_needed=[
            "Detect data drift over time",
            "Visualize model performance",
            "Alert on anomalies",
            "Track system metrics"
        ],
        selected_tools={
            "Grafana": True,
            "Prometheus": True,
            "Evidently": True,
            "Custom dashboards": False
        },
        logo_file="Logo_grafana.png",
        advantages=[
            "Evidently: Deep statistical drift reports",
            "Grafana: Customizable dashboards",
            "Prometheus: Reliable metrics collection",
            "Alert triggering capabilities"
        ],
        disadvantages=[
            "Defining drift baselines requires domain knowledge",
            "Dashboard setup requires configuration"
        ],
        links=[
            {"text": "Grafana (default: admin/admin)", "url": "http://localhost:3000"}
        ],
        screenshot_file="SS_grafana_generic.png"
    )

elif selected_component == "automation":
    render_component_card(
        title="Automation",
        what_needed=[
            "Orchestrate data pipelines",
            "Schedule model retraining",
            "Handle dependencies between tasks",
            "Monitor pipeline execution"
        ],
        selected_tools={
            "Apache Airflow": True,
            "Prefect": False,
            "Dagster": False,
            "Cron": False
        },
        logo_file="Logo_tool_X.png",
        advantages=[
            "Industry standard for workflow orchestration",
            "Rich UI for monitoring",
            "Powerful DAG dependencies",
            "Extensive operator library"
        ],
        disadvantages=[
            "Resource intensive",
            "Complex initial configuration",
            "Learning curve for DAG development"
        ],
        comments=[
            "DAGs: accidents_data_dag, accidents_ml_dag, accidents_dag",
            "Handles ETL, training, and model promotion"
        ],
        links=[
            {"text": "Airflow UI (default: airflow/airflow)", "url": "http://localhost:8080"}
        ]
    )

elif selected_component == "frontend":
    render_component_card(
        title="User Frontend",
        what_needed=[
            "Provide graphical interface for predictions",
            "Present project results",
            "Enable user input for accident data",
            "Display prediction results clearly"
        ],
        selected_tools={
            "Streamlit": True,
            "Gradio": False,
            "FastAPI + HTML": False,
            "PowerPoint": False
        },
        logo_file="Logo_streamlit.png",
        advantages=[
            "Rapid development in Python",
            "Rich widget library",
            "Easy deployment",
            "Built-in session state"
        ],
        disadvantages=[
            "Limited layout flexibility",
            "No native deep linking to sections",
            "Full re-execution on each interaction"
        ],
        comments=[
            "Multi-page app structure for organization",
            "Custom CSS for enhanced styling",
            "Session state for OAuth token persistence"
        ]
    )
