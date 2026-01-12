"""
Streamlit App to present the project work
Search '2CHECK' for points that need to be verified.

!!! Remove the following lines when ready to load real models.
    st.markdown("Testing result display (to remove for production version)")
    result_display(test_pred)

NB for the Readme:

Start the API locally with:
uvicorn src.api.main:app --reload

Start the Streamlit locally with:
PYTHONPATH=. streamlit run src/streamlit/streamlit_app.py

PYTHONPATH might be needed so that the streamlit server finds the class import in utils when not containerized.
"""

import streamlit as st
# import pandas as pd
import datetime as dt
import requests
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional
import os
from dotenv import load_dotenv
from PIL import Image

## Style for a wider display of contents, esp. needed for big schemas.

st.markdown(
    """
    <style>
        .block-container {
            max-width: 90%;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

## Alternatives for max content width
# st.set_page_config(layout="wide")


# Load environment variables -> should not be needed when specifying a .env file in docker-compose, but needed to run without docker
load_dotenv()


### QUICK VARIABLES

api_ip = "http://localhost" # os.environ.get("API_BASE_URL", "http://localhost")
api_port = os.environ.get("API_PORT") # 8000

api_base_address = api_ip + ":" + str(api_port)



### Loading the API class objects

from src.utils.ml_utils import (
    PredictionRequest,
    PredictionResponse
)

### Loading the Mermaid Schemas

from src.utils.schemas import (
    render_mermaid,
    overview_code
)




### PAGE CONTENTS

st.title("Accident Severity Prediction")

st.header("User API Frontend + Components Presentation")
st.sidebar.title("Table of contents")
pages=["Intro",
       "User Frontend",
       "Project Progress",
       "Architecture Overview",
       "MLOps Components",
       "Conclusion"]

page=st.sidebar.radio("Go to", pages)







if page == pages[0]:
    st.subheader('Introduction')
    st.markdown("Welcome to the user frontend of our Accident Severity Prediction Model")
    st.markdown("In the following pages, you'll be able to test our best model and have a closer look" \
    " at the MLOps architecture and its components.")
    st.markdown("To continue, use the menu on the left.")







if page == pages[1]:
    st.subheader('Model Prediction')
    st.markdown("This tool is based on the road accident database available on the Kaggle platform at this address:")
    link_text = "Kaggle: Accidents in France from 2005 to 2016"
    url = "https://www.kaggle.com/datasets/ahmedlahlou/accidents-in-france-from-2005-to-2016"
    st.markdown(f"[{link_text}]({url})")

    st.markdown("In this part you'll be able to obtain severity predictions from the best model.")
    st.markdown("Severity classes are as follow:")
    st.markdown("* 1: Unscathed\n" \
                "* 2: Light injury\n"\
                "* 3: Hospitalized wounded\n"\
                "* 4: Killed")




	### OAuth2 Authentication and Check
    
    st.write("")
    st.write("")
    st.markdown("\n\n##### First you need to identify with the correct credentials:")
    
    api_username = st.text_input("Enter your API Usernane:", "alice")
    api_password = st.text_input("Enter your API Password:", "secret", type="password")
    
    token_url = api_base_address + "/token"
    data = {"username": api_username,
            "password": api_password
            }


    # Debug: display request
    # st.write(request_url)

    if st.button("Identify"):

        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()

            if response.status_code == 200:
                # status = data.get('status', None)  # Use .get to avoid KeyError
                
                token_json = response.json()
                api_token = token_json["access_token"]

                st.write(f"###### Your authentication token:\n{api_token}")
                st.write("<h4 style='color: green;'>Identification successful! You can keep on.</h4>", unsafe_allow_html=True)
                # st.write("###### Identification successful! You can keep on.") # Alternative without html for color

                st.session_state.auth_headers = {"Authorization": f"Bearer {api_token}"}
                
            else:  # Server responded but with an error status
                detail = data.get('detail', None)
                st.write(f"<h4 style='color: red;'>Error: Identification failed! Server returned status code {response.status_code}<br>Detail: {detail}</h4>", unsafe_allow_html=True)
                # st.write(f"###### Error: Server returned status code {response.status_code}\nDetail: {detail}")  # Alternative without html for color

        except requests.ConnectionError:
            st.write("<h4 style='color: red;'>Error: Could not reach the server. Please try again later.</h4>", unsafe_allow_html=True)
            # st.write("###### Error: Could not reach the server. Please try again later.")   # Alternative without html for color
        except requests.Timeout:
            st.write("<h4 style='color: red;'>Error: Request timed out. Please try again.</h4>", unsafe_allow_html=True)
            # st.write("###### Error: Request timed out. Please try again.")  # Alternative without html for color
        except Exception as e:  # Catch any other exceptions
            st.write(f"<h4 style='color: red;'>An unexpected error occurred: <br>{e}</h4>", unsafe_allow_html=True)
            # st.write(f"###### An unexpected error occurred: {e}")  # Alternative without html for color






    ### FEATURES 2CHECK the requested features are not the same as that of the kaggleDB -> to compare with preprocessing

    # instanciating a PredictionRequest object with the random values of the config example (gets directly overwritten by the user input items)

    sfeatures = PredictionRequest(**{
                "year": 2023,
                "month": 6,
                "hour": 14,
                "minute": 30,
                "user_category": 1,
                "sex": 1,
                "year_of_birth": 1990,
                "trip_purpose": 1,
                "security": 1,
                "luminosity": 1,
                "weather": 1,
                "type_of_road": 1,
                "road_surface": 1,
                "latitude": 48.8566,
                "longitude": 2.3522,
                "holiday": 0
                })


    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("\n\n##### Adjust the following features at will "
    "or scroll down to run the model directly.")
    
    # Year of the accident
    sfeatures.year = st.slider("Year of the accident", 2006, 2016, value=2012)

    # Month of the accident
    options = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12}
    smonth = st.select_slider("Month of the accident", list(options.keys()))
    sfeatures.month = options[smonth]

    # Time of the accident
    stime = st.time_input("Time of the accident (24h format)", value = dt.time(14, 00))
    sfeatures.hour = stime.hour
    sfeatures.minute = stime.minute

    # # Alternative time input:
    # shour = st.slider("Hours of the accident time (24h format)", 0, 23, value=13)
    # sminute = st.slider("Minutes of the accident time", 0, 59, value=13)

    # User categories
    options = {
    "Driver": 1,
    "Passenger": 2,
    "Pedestrian": 3,
    "Pedestrian in rollerblade or scooter": 4}
    suser_category = st.selectbox("User Category", list(options.keys()))
    sfeatures.user_category = options[suser_category]

    # User gender
    options = {
    "Male": 1,
    "Female": 2}
    ssex = st.selectbox("User Gender:", list(options.keys()))
    sfeatures.sex = options[ssex]

    sfeatures.year_of_birth = st.slider("Year of birth", 1900, 2026, value=2000)

    # Trip purpose
    options = {
    "Unknown": 0,
    "Work commuting": 1,
    "School commuting": 2,
    "Shopping": 3,
    "Professional use": 4,
    "Leisure": 5,
    "Other": 9}
    strip_purpose = st.selectbox("Trip purpose", list(options.keys()))
    sfeatures.trip_purpose = options[strip_purpose]
    
    # Safety equipment 2CHECK: ARE CATEGORIES RIGHT?
    options = {
    "Unknown": 0,
    "Belt": 1,
    "Helmet": 2,
    "Children's device": 3,
    "Reflective Equipment": 4,
    "Other": 9}
    ssecurity = st.selectbox("Safety equipment", list(options.keys()))
    sfeatures.security = options[ssecurity]

    # Luminosity
    options = {
    "Full day": 1,
    "Twilight or dawn": 2,
    "Night without public lighting": 3,
    "Night with public lighting not lit": 4,
    "Night with public lighting on": 5}
    sluminosity = st.selectbox("Luminosity", list(options.keys()))
    sfeatures.luminosity = options[sluminosity]
    
    # Atmospheric conditions
    options = {
    "Normal": 1,
    "Light rain": 2,
    "Heavy rain": 3,
    "Snow / hail": 4,
    "Fog / smoke": 5,
    "Strong wind / storm": 6,
    "Dazzling weather": 7,
    "Cloudy weather": 8,
    "Other": 9}
    sweather = st.selectbox("Atmospheric conditions", list(options.keys()))
    sfeatures.weather = options[sweather]
    
    # Type of Road
    options = {
    "Highway": 1,
    "National road": 2,
    "Departmental road": 3,
    "Communal way": 4,
    "Off public network": 5,
    "Parking lot open to public traffic": 6,
    "Other": 9}
    stype_of_road = st.selectbox("Type of Road", list(options.keys()))
    sfeatures.type_of_road = options[stype_of_road]

    # Road surface
    options = {
    "Normal": 1,
    "Wet": 2,
    "Puddles": 3,
    "Flooded": 4,
    "Snow": 5,
    "Mud": 6,
    "Icy": 7,
    "Grease / Oil": 8,
    "Other": 9}
    sroad_surface = st.selectbox("Road surface", list(options.keys()))
    sfeatures.road_surface = options[sroad_surface]
    
    sfeatures.latitude = st.number_input("Latitude (format HH.MMMM)", 48.8584)
    
    sfeatures.longitude = st.number_input("Longitude (format HH.MMMM)", 2.2945)

    # Holidays
    options = {
    "No": 0,
    "Yes": 1}
    sholiday = st.radio("Holidays?", list(options.keys()))
    sfeatures.holiday = options[sholiday]

    # Debug
    # st.table(sfeatures)


    # Instanciating a PredictionRequest object from the user inputs

    # sfeatures = PredictionRequest(syear, smonth, shour, sminute, suser_category, ssex, syear_of_birth,
    #                               strip_purpose, ssecurity, sluminosity, sweather, stype_of_road,
    #                               sroad_surface, slatitude, slongitude, sholiday)

    # Converting the PredictionRequest object to a dictionnary with model_dump
    sfeatures_dict = sfeatures.model_dump() # 2CHECK is this necessary if the predict_severity expects a PredictionRequest that it itself converts to dict?





### MODEL EXECUTION


    request_url = api_base_address + "/predict"

    # Debug: display request
    # st.write(request_url)


    def result_display(pred: PredictionResponse):
        st.markdown(f"##### The model predicted the accident severity:\n## {pred.prediction_label}\n##### with a confidence of {pred.confidence}.")
        st.write("")

        if st.checkbox("Show detailed results"):
            data_to_display = {
            "Prediction": test_pred.prediction,
            "Prediction Label": test_pred.prediction_label,
            "Confidence": test_pred.confidence,
            "Model Version": test_pred.model_version
            }

            st.table(data_to_display)

            st.markdown("Probability for each severity class:")
            st.table(pred.probabilities)

    # Dummy PredictionResponse for test purposes
    test_pred = PredictionResponse(**({
                "prediction": 3,
                "prediction_label": "Hospitalized wounded",
                "probabilities": {
                    "Unscathed": 0.1,
                    "Light injury": 0.2,
                    "Hospitalized wounded": 0.65,
                    "Killed": 0.05
                },
                "confidence": 0.65,
                "model_version": "accident_severity_rf_20251210_162637"
            }))

    # # Testing result display 2CHECK
    # st.markdown("Testing result display (to remove for production version)")
    # result_display(test_pred)





    st.write("")
    st.write("")

    if st.button("Get prediction from best available model"):

        try:
            response = requests.post(request_url, json=sfeatures_dict, headers=st.session_state.auth_headers)
            response.raise_for_status()
            data = response.json()

            if response.status_code == 200:

                # Parse the response
                response_data = PredictionResponse(**response.json())
                
                # Display the results
                result_display(response_data)
                # 2CHECK does the actual PredictionResponse get handled and displayed correctly? No model available in the current branch.

                # Untested alternatives without the result_display function
                # Create a DataFrame from the model attributes
                # data_dict = response_data.model_dump()
                # st.table(data_dict) 
                # Or as dataframe:
                # df = pd.DataFrame([data_dict])
                # st.dataframe(df)
            
            else:  # Server responded but with an error status
                st.write(f"<h4 style='color: red;'>Error: Server returned status code {response.status_code}<br>Detail: {response.text}</h4>", unsafe_allow_html=True)
                # st.write(f"###### Error: Server returned status code {response.status_code}\nDetail: {response.text}")   # Alternative without html for color

        except requests.ConnectionError:
            st.write("<h4 style='color: red;'>Error: Could not reach the server. Please try again later.</h4>", unsafe_allow_html=True)
            # st.write("###### Error: Could not reach the server. Please try again later.")   # Alternative without html for color
        except requests.Timeout:
            st.write("<h4 style='color: red;'>Error: Request timed out. Please try again.</h4>", unsafe_allow_html=True)
            # st.write("###### Error: Request timed out. Please try again.")  # Alternative without html for color
        except Exception as e:  # Catch any other exceptions
            st.write(f"<h4 style='color: red;'>An unexpected error occurred: <br>{e}</h4>", unsafe_allow_html=True)
            # st.write(f"###### An unexpected error occurred: {e}")  # Alternative without html for color

    st.write("")
    st.write("")


### GET MODEL INFO

    request_url = api_base_address + "/model/info"

    # Debug: display request
    # st.write(request_url)

    if st.button("Get model information"):

        try:
            response = requests.post(request_url, headers=st.session_state.auth_headers)
            response.raise_for_status()
            data = response.json()

            if response.status_code == 200:

                # Parse the response
                response_data = PredictionResponse(**response.json())
                
                # Create a DataFrame from the model attributes
                data_dict = response_data.model_dump()
                st.markdown("Detailed model information:")
                st.table(data_dict) # 2CHECK does the ModelInfo get handled and displayed correctly? No model available in the current branch.
            
            else:  # Server responded but with an error status
                st.write(f"<h4 style='color: red;'>Error: Server returned status code {response.status_code}<br>Detail: {response.text}</h4>", unsafe_allow_html=True)
                # st.write(f"###### Error: Server returned status code {response.status_code}\nDetail: {response.text}")   # Alternative without html for color

        except requests.ConnectionError:
            st.write("<h4 style='color: red;'>Error: Could not reach the server. Please try again later.</h4>", unsafe_allow_html=True)
            # st.write("###### Error: Could not reach the server. Please try again later.")   # Alternative without html for color
        except requests.Timeout:
            st.write("<h4 style='color: red;'>Error: Request timed out. Please try again.</h4>", unsafe_allow_html=True)
            # st.write("###### Error: Request timed out. Please try again.")  # Alternative without html for color
        except Exception as e:  # Catch any other exceptions
            st.write(f"<h4 style='color: red;'>An unexpected error occurred: <br>{e}</h4>", unsafe_allow_html=True)
            # st.write(f"###### An unexpected error occurred: {e}")  # Alternative without html for color







### PROJECT PROGRESS
# 2CHECK: still needs content, mostly bullet points of the development history + visual contents

if page == pages[2]:

    st.subheader("Project Progress")
    
    st.markdown("Here you'll find content to illustrate our account of the project development.")

    ### Place holder
    # Syntax bullet points
    st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
    # separate for bigger spacing:
    st.markdown("* Blo")

    # Syntax picture display
    #st.image(image, caption=’It’s an image’)




### ARCHITECTURE OVERVIEW


if page == pages[3]:

    st.subheader("Architecture Overview")
    
    st.markdown("Here you can get an overview of the MLOps architecture, " \
    "from data acquisition to model prediction and monitoring.")

    render_mermaid(overview_code)




### COMPONENTS PRESENTATION
# 2CHECK: still needs content

if page == pages[4]:

    st.subheader("MLOps Components")
    
    st.markdown("Here you'll find a detailed description of the various MLOps components we used.")
    st.write("")


    ### construct to enable link access (from overview schema) to selectbox subcategories
    # 2CHECK NOT FUNCTIONAL IN THIS STATE always loads page 0
    # (solution1: components have to go in dedicated pages)
    # (solution2: components have to go in dedicated python files)

    comp_pages = {"Components List":"list", 
                  "Programming Environment":"environment", 
                  "Data Handling":"data",
                  "ML Model":"model",
                  "Prediction API":"api",
                  "Tracking & Versioning":"mlflow",
                  "Containerization":"docker", 
                  "Unit Testing":"tests",
                  "Drift & Retraining":"grafana_evidently",
                  "Automation":"airflow",
                  "User Frontend":"streamlit"}
    
    params = st.query_params
    comp_from_url = params.get("component", "list")

    labels = list(comp_pages.keys())
    values = list(comp_pages.values())

    if comp_from_url in values:
        default_index = values.index(comp_from_url)
    else:
        default_index = 0

    arch_page = st.selectbox(
        "Select a component:",
        labels,
        index=default_index
    )

    page_value = comp_pages[arch_page]

    # keep URL in sync when user clicks selectbox
    st.query_params["page"] = page_value

    
    
    
    if arch_page == "Components List" :
        st.subheader("Components List")

        st.markdown("##### We sorted the various tools used in the following categories:")
        
        only_comps = list(comp_pages.keys())[1:]
        st.write("")
        i = 1
        for key in only_comps:
            st.markdown(f"* {key}")
            i += 1
        st.write("")


    
    
    elif arch_page == "Programming Environment" :
        st.subheader("Programming Environment") #(GitHub Project +) Repro env

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Split the project in subtasks\n" \
                "* Identify subtasks order and dependencies\n"\
                "* Coordinate the work of several people with a realistic timeline\n"
                "* Safely track each step the development in a centralized manner")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("GitLab", value=False)
        st.checkbox("GitHub + GitHub Projects", value=True)
        st.checkbox("Codeberg", value=False)
        st.checkbox("Gitea", value=False)

        logoX = Image.open("src/streamlit/Logo_github.png") # Edit the source file name
        st.image(logoX, width=300)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Reliable widespread Git-based tool\n" \
                "* Code can be viewed and edited on the online platform\n" \
                "* Project management directly integrated in the repository\n"\
                "* Easy Kanban-style task management")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Complexity / learning curve\n" \
                "* Limited advanced reporting")
        st.write("")
        
        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* The base repository had to be forked to our own GitHub account to have access to the Project functions.")
        
        screenshotX = Image.open("src/streamlit/SS_github_project.png") # Edit the source file name
        st.image(screenshotX, caption="Screenshot of GitHub Project", width=1000)
        st.write("")
        st.write("")
        st.write("")





        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("Ensure a reproducible work environment: \n" \
                "* ...for each team member\n"\
                "* ...independently of OS-platforms, IDEs and last updates\n" \
                "* Convenient and fast setup")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("Conda", value=False)
        st.checkbox("Poetry", value=False)
        st.checkbox("Pipenv", value=False)
        st.checkbox("Astral UV", value=True)

        logoX = Image.open("src/streamlit/Logo_uv.jpg") # Edit the source file name
        st.image(logoX, width=300)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Cross-platform tool to replace pip, pip-tools, pipx, virtualenv...\n" \
                "* Extremely fast (10-100x faster than pip)\n" \
                "* Installs/adapts both the required virtual python environment and its dependencies\n"\
                "* Disk-space efficient, with a global cache for dependencies\n" \
                "* Exact reproducibility (local dev / Docker containers) with uv.lock file")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Needs prior installation\n" \
                "* Relatively new tool with its own syntax\n"\
                "* Some rare packages might need specific configuration.")
        st.write("")

        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Reduced environment setup time from minutes to seconds!")
        
        # [Optional] Syntax of a link to the actual tool interface
        link_text = "To the UV website" # Ex: to the airflow live interface 
        url = "https://docs.astral.sh/uv/"
        st.markdown(f"[{link_text}]({url})")
        
        screenshotX = Image.open("src/streamlit/SS_uv.png") # Edit the source file name
        st.image(screenshotX, caption="Screenshot of .venv setup with UV")
        st.write("")





    elif arch_page == "Data Handling" :
        st.subheader("Data Handling") # SQL DB (+ Daten aus Kaggle + Preprocess)

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("NoSQL", value=False)
        st.checkbox("PostgreSQL", value=True)
        st.checkbox("SQLight", value=False)

        logoX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(logoX, caption="Logo of tool X", width=300)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        
        # [Optional] Syntax of a link to the actual tool interface
        link_text = "To the actual tool live interface" # Ex: to the airflow live interface 
        url = "http://airflow:8000"
        st.markdown(f"[{link_text}]({url})")
        
        screenshotX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(screenshotX, caption="Screenshot of tool X", width=1000)
        st.write("")





    elif arch_page == "ML Model" :
        st.subheader("ML Model") # ML Model (choice, training, validation)

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("NoSQL", value=False)
        st.checkbox("PostgreSQL", value=True)
        st.checkbox("SQLight", value=False)

        logoX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(logoX, caption="Logo of tool X", width=300)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        
        # [Optional] Syntax of a link to the actual tool interface
        link_text = "To the actual tool live interface" # Ex: to the airflow live interface 
        url = "http://airflow:8000"
        st.markdown(f"[{link_text}]({url})")
        
        screenshotX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(screenshotX, caption="Screenshot of tool X", width=1000)
        st.write("")





    elif arch_page == "Prediction API" :
        st.subheader("Prediction API") # API + OAuth2

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("NoSQL", value=False)
        st.checkbox("PostgreSQL", value=True)
        st.checkbox("SQLight", value=False)

        logoX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(logoX, caption="Logo of tool X", width=300)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        
        # [Optional] Syntax of a link to the actual tool interface
        link_text = "To the actual tool live interface" # Ex: to the airflow live interface 
        url = "http://airflow:8000"
        st.markdown(f"[{link_text}]({url})")
        
        screenshotX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(screenshotX, caption="Screenshot of tool X", width=1000)
        st.write("")




    elif arch_page == "Tracking & Versioning" :
        st.subheader("Tracking & Versioning") # MLFlow + MinIO + Tag bestes Model

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("NoSQL", value=False)
        st.checkbox("PostgreSQL", value=True)
        st.checkbox("SQLight", value=False)

        logoX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(logoX, caption="Logo of tool X", width=300)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        
        # [Optional] Syntax of a link to the actual tool interface
        link_text = "To the MLFlow interface (default credentials: mlflow / mlflow)" # Ex: to the airflow live interface 
        url = "http://localhost:5001"
        st.markdown(f"[{link_text}]({url})")

        link_text = "To the MinIO interface (default credentials: mini_user / mini_password)" # Ex: to the airflow live interface 
        url = "http://localhost:9000"
        st.markdown(f"[{link_text}]({url})")
        
        screenshotX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(screenshotX, caption="Screenshot of tool X", width=1000)
        st.write("")





    elif arch_page == "Containerization" :
        st.subheader("Containerization") # Docker

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Package all services into portable units that can run consistently " \
        "across any environment (cloud or local).\n"\
            "* Orchestrate multiple containers (API, DB, MLFlow...) to work together.")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("Docker (with Docker Compose)", value=True)
        st.checkbox("Podman", value=False)
        st.checkbox("Firecracker", value=False)
        st.checkbox("Containerd (runtime) + Kubernetes (orchestration)", value=False)
        st.checkbox("Bare metal deployment / manual scripting", value=False)


        logoX = Image.open("src/streamlit/Logo_docker.jpg") # Edit the source file name
        st.image(logoX, width=300)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Widespread cross-platform tool (-> rich DockerHub)\n" \
                "* Precise networking and volume management\n"\
                "* Docker Compose simplifies multi-service orchestration (single command)")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Long build process, heavy disk and RAM usage if not optimized\n" \
                "* [Hard build errors debugging because of CRLF EOL]\n"\
                "* [Dockerfile copy syntax slighty differs from standard shell commands]")
        st.write("")

        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("Even after build optimization, the entirety of the architecture including airflow was too RAM " \
        "intensive to run on the Datascientest computer. The error messages were often opaque or misleading.")
        
        # [Optional] Syntax of a link to the actual tool interface
        link_text = "To the Docker website" # Ex: to the airflow live interface 
        url = "https://www.docker.com/"
        st.markdown(f"[{link_text}]({url})")
        
        screenshotX = Image.open("src/streamlit/SS_docker.png") # Edit the source file name
        st.image(screenshotX, caption="Screenshot of Docker Desktop")
        st.write("")





    elif arch_page == "Unit Testing" :
        st.subheader("Unit Testing") # unit testing (evtl. + GitHub Actions)

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Implement unit tests (+ linter) to ensure the different services operate properly.\n")
        st.write("")

        st.markdown("##### Which tool was selected") # 2CHECK how are the unit tests ran and how is linting done. Any GitHub actions?
        st.checkbox("Manual testing", value=False)
        st.checkbox("Pytest + Unit tests files", value=True)
        st.checkbox("GitHub Actions + Yaml file", value=False)
        st.checkbox("VSCode standard linting extensions (Flake8, Black...)", value=True)

        logoX = Image.open("src/streamlit/Logo_pytest.png") # Edit the source file name
        st.image(logoX, width=200)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Pytest: each service gets tested with precise log output\n" \
                "* Extensions: Linting is done on the fly while coding")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Only python and bash files get linted\n" \
                "* Manual execution through make commands, no automation on push like GitHub Actions")
        st.write("")

        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Not enough time to implement CI/CD process in GitHub Actions.")
        
        # # [Optional] Syntax of a link to the actual tool interface
        # link_text = "To the actual tool live interface" # Ex: to the airflow live interface 
        # url = "http://airflow:8000"
        # st.markdown(f"[{link_text}]({url})")
        
        screenshotX = Image.open("src/streamlit/SS_testing.png") # Edit the source file name
        st.image(screenshotX, caption="Screenshot unit tests", width=600)
        st.write("")




    elif arch_page == "Drift & Retraining" : # whole section 2CHECK
        st.subheader("Drift & Retraining") # Grafana + Evidently

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Monitor the incoming data for 'Data Drift' (changes in statistical properties)\n" \
                "* Visualize model performance over time")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("Alchemy", value=False)
        st.checkbox("Crossbow", value=False)
        st.checkbox("Evidently", value=True)
        st.checkbox("Grafana", value=True)
        st.checkbox("Kite", value=False)

        logoX = Image.open("src/streamlit/Logo_grafana.png") # Edit the source file name
        st.image(logoX, width=300)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Evidently: provides deep statistical reports on data drift\n" \
                "* Grafana: access restricted web based visualization tool with customizable dashboards\n"\
                "* Grafana: event alerts and action triggering")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Not easy to define proper references to detect drifts") 
        st.write("")

        # st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        # st.markdown("* Bla\n" \
        #         "* Bli\n"\
        #         "* Blu")
        
        # [Optional] Syntax of a link to the actual tool interface
        link_text = "To the actual Grafana web interface (default credentials: admin / admin)" # Ex: to the airflow live interface 
        url = "http://localhost:3000 "
        st.markdown(f"[{link_text}]({url})")
        
        screenshotX = Image.open("src/streamlit/SS_grafana_generic.png") # 2CHECK add real user case screenshot
        st.image(screenshotX, caption="Grafana UI", width=500) 
        st.write("")





    elif arch_page == "Automation" :
        st.subheader("Automation") # Airflow

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("NoSQL", value=False)
        st.checkbox("PostgreSQL", value=True)
        st.checkbox("SQLight", value=False)

        logoX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(logoX, caption="Logo of tool X", width=300)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Accidents data dag: Manages chunked or full data ingestion. \
                    This allows for simulating data evolution over time.\n" \
                "* Accidents ml dag: Handles the machine learning pipeline, including data cleaning, \
                    dataset splitting, model training and evaluation.\n"\
                "* Accidents dag: Orchestrates the ETL pipeline for data ingestion, cleaning, and model training.")
        st.write("")

        st.markdown("The project includes the following Airflow DAGs for orchestrating workflows:") # Ex: structured data storing
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        st.write("")

        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Bla\n" \
                "* Bli\n"\
                "* Blu")
        
        # [Optional] Syntax of a link to the actual tool interface
        link_text = "To the Airflow live interface (default credentials: airflow / airflow)" # Ex: to the airflow live interface 
        url = "http://localhost:8080"
        st.markdown(f"[{link_text}]({url})")
        
        screenshotX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
        st.image(screenshotX, caption="Screenshot of tool X", width=1000)
        st.write("")





    elif arch_page == "User Frontend" :
        st.subheader("User Frontend") # Streamlit

        ### Layout for standard component description

        st.markdown("##### What needed to be done") # Ex: store the accident data
        st.markdown("* Provide a reliable and lightweight graphical interface to present results\n" \
                "* Provide an interactive API frontend for users to input accident data and view predictions.")
        st.write("")

        st.markdown("##### Which tool was selected")
        st.checkbox("Google Slides / Powerpoint", value=False)
        st.checkbox("Streamlit", value=True)
        st.checkbox("FastAPI Docs / Swagger", value=False)

        logoX = Image.open("src/streamlit/Logo_streamlit.png") # Edit the source file name
        st.image(logoX, width=500)
        st.write("")

        st.markdown("##### Advantages") # Ex: structured data storing
        st.markdown("* Fast development in Python -> possibility to connect external content\n" \
                "* Good widgets for user input and data visualization\n"\
                "* Lightweight webpage with simple interface")
        st.write("")

        st.markdown("##### Disadvantages [/ Issues]") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Limited layout flexibility compared to custom CSS\n" \
                "* No effective way to weblink to specific content within pages")
        st.write("")

        st.markdown("##### Comments on the results") # Ex: difficult DB initialization with persistence in Docker
        st.markdown("* Used to build both the user prediction app and the project presentation itself\n" \
                "* As each new user input triggers a complete topdown execution of the streamlit code, it needs \
                special care to keep variables (like the API OAuth2 token) in memory")
        






### CONCLUSION AND OUTLOOK
# 2CHECK: still needs content, mostly bullet points

if page == pages[5]:

    st.subheader("Conclusion and Outlook") #2CHECK
    

    ### Place holder
    # Syntax bullet points
    st.markdown("* Setup a reproducible development environment and a project organization for the team\n" \
                "* Devised a complete MLOps architecture, from data handling to model training and API frontend for user inference.\n"\
                "* The correct execution can be tested during development and performance and operations monitored\n"\
                "* The entirety of the architecture can be easily deployed in one command as an orchestrated group of containerized microservices")
    st.write("")
    # separate for bigger spacing:
    st.markdown("* Several improvements remain to be tested/implemented (Kubernetes scaling, GitHub Actions...) as health issues\
                 in the team strongly reduced the available development time")
    st.write("")
    st.markdown("##### Thank you!")
    # # Syntax picture display
    # screenshotX = Image.open("src/streamlit/Logo_tool_X.png") # Edit the source file name
    # st.image(screenshotX, caption="Screenshot of tool X", width=1000)
    # st.write("")