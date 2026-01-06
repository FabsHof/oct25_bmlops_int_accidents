"""
Streamlit App to present the project work
Search '2CHECK' for points that need to be verified.

NB for the Readme:

Start the API locally with:
uvicorn src.api.main:app --reload

Start the Streamlit locally with:
streamlit run src/streamlit/streamlit_app.py
"""
import streamlit as st
# import pandas as pd
import datetime as dt
import requests
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional

### QUICK VARIABLES

api_ip = "127.0.0.1"
api_port = 8000

api_base_address = api_ip + ":" + str(api_port)



### Loading the API class objects

from src.utils.model_utils import (
    PredictionRequest,
    PredictionResponse
)


### Deprecated because imported from utils above: Pydantic models for request/response validation
# class PredictionRequest(BaseModel):
#     """Request model for accident severity prediction."""
#     year: int = Field(..., description="Year of the accident", ge=2000, le=2100)
#     month: int = Field(..., description="Month of the accident", ge=1, le=12)
#     hour: int = Field(..., description="Hour of the accident", ge=0, le=23)
#     minute: int = Field(..., description="Minute of the accident", ge=0, le=59)
#     user_category: int = Field(..., description="User category (e.g., driver, passenger, pedestrian)")
#     sex: int = Field(..., description="Sex of the user")
#     year_of_birth: int = Field(..., description="Year of birth of the user", ge=1900, le=2100)
#     trip_purpose: int = Field(..., description="Purpose of the trip")
#     security: int = Field(..., description="Security equipment used")
#     luminosity: int = Field(..., description="Luminosity conditions")
#     weather: int = Field(..., description="Weather conditions")
#     type_of_road: int = Field(..., description="Type of road")
#     road_surface: int = Field(..., description="Road surface condition")
#     latitude: float = Field(..., description="Latitude of the accident location")
#     longitude: float = Field(..., description="Longitude of the accident location")
#     holiday: int = Field(..., description="Holiday indicator (0 or 1)", ge=0, le=1)
    
#     model_config = ConfigDict(
#         json_schema_extra={
#             "example": {
#                 "year": 2023,
#                 "month": 6,
#                 "hour": 14,
#                 "minute": 30,
#                 "user_category": 1,
#                 "sex": 1,
#                 "year_of_birth": 1990,
#                 "trip_purpose": 1,
#                 "security": 1,
#                 "luminosity": 1,
#                 "weather": 1,
#                 "type_of_road": 1,
#                 "road_surface": 1,
#                 "latitude": 48.8566,
#                 "longitude": 2.3522,
#                 "holiday": 0
#             }
#         }
#     )


# class PredictionResponse(BaseModel):
#     """Response model for accident severity prediction."""
#     prediction: int = Field(..., description="Predicted severity class (1-4)")
#     prediction_label: str = Field(..., description="Human-readable severity label")
#     probabilities: Dict[str, float] = Field(..., description="Probability for each severity class")
#     confidence: float = Field(..., description="Confidence score (max probability)")
#     model_version: Optional[str] = Field(None, description="Version of the model used")
    
#     model_config = ConfigDict(
#         json_schema_extra={
#             "example": {
#                 "prediction": 3,
#                 "prediction_label": "Hospitalized wounded",
#                 "probabilities": {
#                     "Unscathed": 0.1,
#                     "Light injury": 0.2,
#                     "Hospitalized wounded": 0.65,
#                     "Killed": 0.05
#                 },
#                 "confidence": 0.65,
#                 "model_version": "accident_severity_rf_20251210_162637"
#             }
#         }
#     )




### PAGE CONTENTS

st.title("Accident Severity Prediction")

st.header("User API Frontend + Components Presentation")
st.sidebar.title("Table of contents")
pages=["Intro",
       "User Frontend",
       "MLOps Architecture"]

page=st.sidebar.radio("Go to", pages)







if page == pages[0]:
    st.subheader('Introduction')
    st.markdown("Welcome to the user frontend of our Accident Severity Prediction Model")
    st.markdown("In the following pages, you'll be able to test our best model and have a closer look" \
    " at the MLOps architecture and its components.")
    st.markdown("To continue, use the menu on the left.")







if page == pages[1]:
    st.subheader('Model Prediction')
    st.markdown("In this part you'll be able to obtain severity predictions from the best model.")
    st.markdown("Severity classes are as follow:")
    st.markdown("* 1: Unscathed\n" \
                "* 2: Light injury\n"\
                "* 3: Hospitalized wounded\n"\
                "* 4: Killed")




    ### API Key and API status check
    
    st.markdown("\n\n##### First you need to identify with the correct API Key:")
    
    api_key = st.text_input("Enter your API Key:", "secret", type="password")
    request_url = "http://" + api_base_address + "/health" + "?api_key=" + api_key

    # Debug: display request
    # st.write(request_url)

    if st.button("Identify"):

        try:
            response = requests.get(request_url)
            data = response.json()

            if response.status_code == 200:
                status = data.get('status', None)  # Use .get to avoid KeyError

                if status == "healthy":
                    st.write("<h4 style='color: green;'>Identification successful! You can keep on.</h4>", unsafe_allow_html=True)
                    # st.write("###### Identification successful! You can keep on.") # Alternative without html for color

                else:  # Identification unsuccessful but server reached
                    st.write(f"<h4 style='color: red;'>Identification failed!<br>Error Code: {status}</h4>", unsafe_allow_html=True)
                    # st.write(f"###### Identification failed!\nError Code: {status}")  # Alternative without html for color
            
            else:  # Server responded but with an error status
                detail = data.get('detail', None)
                st.write(f"<h4 style='color: red;'>Error: Server returned status code {response.status_code}<br>Detail: {detail}</h4>", unsafe_allow_html=True)
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






    ### Features

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


    request_url = "http://" + api_base_address + "/predict" + "?api_key=" + api_key

    #     2CHECK: it might be better to transmit the api_key in the headers? not sure how exactly.
    #     headers = {"api_key":api_key,
    #               # "Authorization": f"Bearer {api_key}",
    #               "Content-Type": "application/json"}

    # Debug: display request
    # st.write(request_url)





    def result_display(pred: PredictionResponse):
        st.markdown(f"##### The model predicted the accident severity:\n## {pred.prediction_label}\n##### with a confidence of {pred.confidence}.")

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

    # Testing result display 2CHECK
    st.markdown("Testing result display (to remove for production version)")
    result_display(test_pred)







    if st.button("Get prediction from best available model"):

        try:
            response = requests.post(request_url, json=sfeatures_dict)#, headers=headers)
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



### GET MODEL INFO

    request_url = "http://" + api_base_address + "/model/info" + "?api_key=" + api_key

    #     2CHECK: it might be better to transmit the api_key in the headers? not sure how exactly.
    #     headers = {"api_key":api_key,
    #               # "Authorization": f"Bearer {api_key}",
    #               "Content-Type": "application/json"}

    # Debug: display request
    # st.write(request_url)

    if st.button("Get model information"):

        try:
            response = requests.get(request_url)
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




### 2CHECK: Place holder optional: 
# - ingest data chunk
# - train model
# - get and reset ingestion progress







### ARCHITECTURE AND COMPONENTS PRESENTATION
# 2CHECK: still needs content

if page == pages[2]:

    st.subheader("MLOps Architecture")
    
    st.markdown("Here you can get an overview of the MLOps architecture, " \
    "from data acquisition to model prediction")

    # return to the overview when going back to page 2
    arch_page="Architecture Overview" 

    arch_page = st.selectbox("Select a component:", 
                             ("Architecture Overview", "ETL Process",
                              "Train and Evaluate Model",
                              "Determine Production Model",
                              "Prediction API"))
    
    if arch_page == "Architecture Overview" :
        st.subheader("Architecture Overview")

    if arch_page == "ETL Process" :
        st.subheader("ETL Process")

    if arch_page == "Train and Evaluate Model" :
        st.subheader("Train and Evaluate Model")
        
    if arch_page == "Determine Production Model" :
        st.subheader("Determine Production Model")
        
    if arch_page == "Prediction API" :
        st.subheader("Prediction API")
