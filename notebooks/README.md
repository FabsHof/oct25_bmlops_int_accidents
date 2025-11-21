# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis, model development, and experimentation.

## Available Notebooks

### predictions-of-severity.ipynb

**Description**: A comprehensive notebook for predicting the severity of road accidents in France using machine learning models.

**Features**:
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Multiple ML models: Random Forest, Logistic Regression, Decision Tree
- Model evaluation and comparison
- Feature importance analysis

**Dataset**: Accidents in France from 2005 to 2016

**How to use**:
1. Install required dependencies (they should already be in the project's `pyproject.toml`)
2. Download the dataset using kagglehub:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("ahmedlahlou/accidents-in-france-from-2005-to-2016")
   ```
3. Load your data and run the pipeline function:
   ```python
   import pandas as pd
   df = pd.read_csv('path/to/your/data.csv')
   results = run_severity_prediction_pipeline(df, target_col='grav')
   ```

**Key Functions**:
- `preprocess_data()`: Handles missing values and data cleaning
- `prepare_features()`: Encodes categorical variables and scales features
- `train_random_forest()`: Trains a Random Forest classifier
- `train_logistic_regression()`: Trains a Logistic Regression model
- `train_decision_tree()`: Trains a Decision Tree classifier
- `evaluate_model()`: Evaluates model performance with metrics and visualizations
- `plot_feature_importance()`: Visualizes the most important features
- `run_severity_prediction_pipeline()`: Complete end-to-end pipeline

**Requirements**:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
