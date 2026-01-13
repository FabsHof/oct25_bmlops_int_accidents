"""
Function to render mermaid.js code in Streamlit
+
Streamlit Code of the architecture overview (for streamlit and README.md)
"""

import streamlit as st
import streamlit.components.v1 as components


def render_mermaid(code):
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    <pre class="mermaid">{code}</pre>
    <script>
      mermaid.initialize({{ startOnLoad: true }});
    </script>
    """
    components.html(html, height=1000, width=2800, scrolling=False)




# Architecture Overview

overview_code = """

%% %%{init: {"flowchart": {"curve": "curve"}}}%%
%% Choose between curve, linear, step, cardinal
%% default: curve

flowchart LR

    classDef transp fill:transparent,stroke:transparent;
    classDef user fill:#ff6f00,stroke:#b34700,stroke-width:3px,color:#ffffff;
    classDef airflow fill:#ffd54f,stroke:#b28704,stroke-width:3px,color:#000000;
    classDef api fill:#00c853,stroke:#007e33,stroke-width:3px,color:#ffffff;
    classDef db fill:#2962ff,stroke:#0039cb,stroke-width:3px,color:#ffffff;
    classDef mlflow fill:#00b0ff,stroke:#007bb2,stroke-width:3px,color:#ffffff;

    linkStyle default stroke:#000999,stroke-width:2px

    subgraph UA[USER APP]
        IFS[INTERACTIVE FEATURE INPUT]:::user
    end

    IFS --> |Features Input|EP



    subgraph MLF[MLFLOW]
        SR[STORE RUN]:::mlflow
        PLMS[PROD & LAST MODEL SCORE]:::mlflow
        UT[UPDATE TAGS]:::mlflow
        IPM[IDENTIFY PROD. MODEL]:::mlflow
    end



    IPM --> |Prod. Model|EP
    PLMS --> |Metrics|DPM





    subgraph API[MODEL API]
        ET[ENDPOINT /train]:::api
        EP[ENDPOINT /predict]:::api
    end


    EP --> |Query Prod. Model|IPM
    EP --> |Prediction|IFS
    ET -->|Data Query|FPD
    ET -->|Last Model & Metrics|SR



    subgraph DB[DATABASE]
        SRD[STORE RAW DATA]:::db
        SPD[STORE PREPROCESSED DATA]:::db
        FPD[FETCH PREPROCESSED DATA]:::db
    end

    FPD -->|Preprocessed Data|ET



    subgraph CAF[AIRFLOW]
        START[PROCESS START]:::airflow
        ETL[ETL]:::airflow
        TE[TRAIN & EVALUATE]:::airflow
        DPM[IDENTIFY PROD. MODEL]:::airflow
        END[PROCESS END<br>/ LOOP TO ETL]:::airflow
    end

    START --> ETL --> TE --> DPM --> END
    DPM -->|IF last model is better: Update Query|UT
    DPM --> |Last & Prod. Model Score Query|PLMS
    ETL --> |csv Files Query|DD
    ETL --> |Raw Data|SRD
    ETL --> |Preprocessed Data|SPD
    TE --> |Train New Model Query|ET
    linkStyle 8 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5
    linkStyle 9 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5
    linkStyle 10 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5
    linkStyle 11 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5

    

    subgraph KG[KAGGLE]
        DD[DATASET DOWNLOAD]:::db
    end

    DD --> |Raw csv Files|ETL



    subgraph LG[LEGEND]
        direction LR
        PROCESS
        D1[ ]:::transp
        D2[ ]:::transp
        D3[ ]:::transp
        D4[ ]:::transp
        D1 --> |DATA| D2
        D3 --> |PROCESS STEPS| D4
        linkStyle 20 stroke:#BA8E23,stroke-width:6px,stroke-dasharray:5 5
    end

"""