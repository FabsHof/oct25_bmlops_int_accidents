"""
Accident Severity Prediction - Main Application

A Streamlit application for French police departments to predict
the severity of traffic accidents based on various parameters.

Run locally with:
    PYTHONPATH=. streamlit run src/streamlit/Home.py

Run with Docker:
    docker-compose up streamlit
"""

import streamlit as st

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": "# Accident Severity Prediction\nA tool for French police departments."
    }
)

# Custom CSS for consistent styling
st.markdown(
    """
    <style>
        /* Main container width */
        .block-container {
            max-width: 90%;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f0f4f8;
        }
        
        /* Header styling */
        h1 {
            color: #1E3A5F;
            border-bottom: 3px solid #1E3A5F;
            padding-bottom: 0.5rem;
        }
        
        /* Card-like containers */
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 0.5rem;
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.image("assets/images/Logo_streamlit.png", width=200) if False else None
        st.markdown("## 🚗 Navigation")
        st.markdown("---")
        st.markdown(
            """
            **Quick Links:**
            - 🏠 Home
            - 🎯 [Prediction](/Prediction)
            - 📈 [Project Progress](/Project_Progress)
            - 🏗️ [Architecture](/Architecture)
            - 🧩 [Components](/Components)
            - 📝 [Conclusion](/Conclusion)
            """
        )
        st.markdown("---")
        st.caption("© 2026 Accident Severity Prediction")
    
    # Main content
    st.title("🚗 Accident Severity Prediction")
    st.markdown("### User API Frontend & Components Presentation")
    
    st.markdown("---")
    
    # Welcome section with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
            ## Welcome
            
            This application provides French police departments with a powerful tool 
            to predict the severity of traffic accidents using machine learning.
            
            ### What you can do:
            
            - **🎯 Make Predictions**: Input accident parameters and get severity predictions
            - **📊 View Architecture**: Explore our MLOps infrastructure
            - **🧩 Learn Components**: Understand the tools and technologies used
            - **📈 Track Progress**: See how the project was developed
            """
        )
        
        st.info(
            "👈 Use the sidebar to navigate between pages, or click the cards below."
        )
    
    with col2:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1E3A5F, #3d5a80);
                color: white;
                padding: 2rem;
                border-radius: 1rem;
                text-align: center;
            ">
                <h2 style="color: white; margin: 0;">🇫🇷</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Based on French road accident data from 2005-2016
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Quick access cards
    st.markdown("### Quick Access")
    
    cols = st.columns(4)
    
    cards = [
        {
            "icon": "🎯",
            "title": "Prediction",
            "description": "Get severity predictions",
            "page": "Prediction"
        },
        {
            "icon": "🏗️",
            "title": "Architecture",
            "description": "View MLOps overview",
            "page": "Architecture"
        },
        {
            "icon": "🧩",
            "title": "Components",
            "description": "Explore our tools",
            "page": "Components"
        },
        {
            "icon": "📈",
            "title": "Progress",
            "description": "Project development",
            "page": "Project_Progress"
        },
    ]
    
    for idx, card in enumerate(cards):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 0.75rem;
                    padding: 1.5rem;
                    text-align: center;
                    height: 150px;
                    transition: transform 0.2s;
                ">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{card['icon']}</div>
                    <div style="font-weight: bold; font-size: 1.1rem; color: #1E3A5F;">
                        {card['title']}
                    </div>
                    <div style="font-size: 0.85rem; color: #666;">
                        {card['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    # Data source info
    with st.expander("ℹ️ About the Data Source"):
        st.markdown(
            """
            This prediction model is based on the road accident database available on Kaggle:
            
            **[Accidents in France from 2005 to 2016](https://www.kaggle.com/datasets/ahmedlahlou/accidents-in-france-from-2005-to-2016)**
            
            The dataset contains detailed information about road accidents in France, 
            including weather conditions, road types, user characteristics, and outcomes.
            """
        )


if __name__ == "__main__":
    main()
