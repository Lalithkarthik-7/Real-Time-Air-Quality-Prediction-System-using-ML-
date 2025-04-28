import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
import requests

# Page configuration
st.set_page_config(page_title="AQI Analysis", page_icon="🌍", layout="wide")

# Title
st.title("Air Quality Index (AQI) Analysis and Prediction")
st.markdown("""
This app allows you to analyze and predict the Air Quality Index (AQI) based on pollutant levels.
Upload your dataset, explore visualizations, and train machine learning models.
""")

# Sidebar for user inputs
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='unicode_escape')
        st.sidebar.success("Dataset uploaded successfully!")
        
        # # Display dataset - ONLY AFTER df IS SUCCESSFULLY CREATED
        # st.subheader("Dataset Preview")
        # st.write(df.head())
        
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.stop()  # This will stop the app execution if there's an error
else:
    st.sidebar.info("Please upload a CSV file to proceed.")
    st.stop()


# Display dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Data Cleaning
st.subheader("Data Cleaning")
st.write("Handling missing values and dropping unnecessary columns.")

# Make a copy of the original dataframe for cleaning
df_clean = df.copy()

# Drop unnecessary columns if they exist
columns_to_drop = ['agency', 'stn_code', 'date', 'sampling_date', 'location_monitoring_station']
columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
df_clean.drop(columns_to_drop, axis=1, inplace=True)

# Fill missing values
if 'location' in df_clean.columns:
    df_clean['location'] = df_clean['location'].fillna(df_clean['location'].mode()[0])
if 'type' in df_clean.columns:
    df_clean['type'] = df_clean['type'].fillna(df_clean['type'].mode()[0])
    
# Fill remaining numerical columns with 0
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numerical_cols] = df_clean[numerical_cols].fillna(0)

st.write("Missing values after cleaning:")
st.write(df_clean.isnull().sum())

# Feature Engineering
st.subheader("Feature Engineering")
st.write("Calculating sub-indices for pollutants and AQI.")

def cal_SOi(so2):
    if so2 <= 40:
        return so2 * (50 / 40)
    elif 40 < so2 <= 80:
        return 50 + (so2 - 40) * (50 / 40)
    elif 80 < so2 <= 380:
        return 100 + (so2 - 80) * (100 / 300)
    elif 380 < so2 <= 800:
        return 200 + (so2 - 380) * (100 / 420)
    elif 800 < so2 <= 1600:
        return 300 + (so2 - 800) * (100 / 800)
    else:
        return 400 + (so2 - 1600) * (100 / 800)

if 'so2' in df_clean.columns:
    df_clean['SOi'] = df_clean['so2'].apply(cal_SOi)

def cal_Noi(no2):
    if no2 <= 40:
        return no2 * 50 / 40
    elif 40 < no2 <= 80:
        return 50 + (no2 - 40) * (50 / 40)
    elif 80 < no2 <= 180:
        return 100 + (no2 - 80) * (100 / 100)
    elif 180 < no2 <= 280:
        return 200 + (no2 - 180) * (100 / 100)
    elif 280 < no2 <= 400:
        return 300 + (no2 - 280) * (100 / 120)
    else:
        return 400 + (no2 - 400) * (100 / 120)

if 'no2' in df_clean.columns:
    df_clean['Noi'] = df_clean['no2'].apply(cal_Noi)

def cal_RSPMI(rspm):
    if rspm <= 30:
        return rspm * 50 / 30
    elif 30 < rspm <= 60:
        return 50 + (rspm - 30) * 50 / 30
    elif 60 < rspm <= 90:
        return 100 + (rspm - 60) * 100 / 30
    elif 90 < rspm <= 120:
        return 200 + (rspm - 90) * 100 / 30
    elif 120 < rspm <= 250:
        return 300 + (rspm - 120) * (100 / 130)
    else:
        return 400 + (rspm - 250) * (100 / 130)

if 'rspm' in df_clean.columns:
    df_clean['Rpi'] = df_clean['rspm'].apply(cal_RSPMI)

def cal_SPMi(spm):
    if spm <= 50:
        return spm * 50 / 50
    elif 50 < spm <= 100:
        return 50 + (spm - 50) * (50 / 50)
    elif 100 < spm <= 250:
        return 100 + (spm - 100) * (100 / 150)
    elif 250 < spm <= 350:
        return 200 + (spm - 250) * (100 / 100)
    elif 350 < spm <= 430:
        return 300 + (spm - 350) * (100 / 80)
    else:
        return 400 + (spm - 430) * (100 / 430)

if 'spm' in df_clean.columns:
    df_clean['SPMi'] = df_clean['spm'].apply(cal_SPMi)

def cal_aqi(row):
    pollutants = []
    if 'SOi' in row: pollutants.append(row['SOi'])
    if 'Noi' in row: pollutants.append(row['Noi'])
    if 'Rpi' in row: pollutants.append(row['Rpi'])
    if 'SPMi' in row: pollutants.append(row['SPMi'])
    return max(pollutants) if pollutants else np.nan

df_clean['AQI'] = df_clean.apply(cal_aqi, axis=1)

def AQI_Range(x):
    if x <= 50:
        return "Good"
    elif 50 < x <= 100:
        return "Moderate"
    elif 100 < x <= 200:
        return "Poor"
    elif 200 < x <= 300:
        return "Unhealthy"
    elif 300 < x <= 400:
        return "Very Unhealthy"
    else:
        return "Hazardous"

if 'AQI' in df_clean.columns:
    df_clean['AQI_Range'] = df_clean['AQI'].apply(AQI_Range)

st.write("Dataset after feature engineering:")
st.write(df_clean.head())

# Data Visualization
st.subheader("Data Visualization")

# State-wise AQI
if 'state' in df_clean.columns and 'AQI' in df_clean.columns:
    st.write("State-wise AQI Distribution")
    plt.figure(figsize=(15, 6))
    sns.barplot(x='state', y='AQI', data=df_clean)
    plt.xticks(rotation=90)
    plt.tight_layout()  # Prevent label cutoff
    st.pyplot(plt)
    plt.clf()  # Clear the figure to prevent overlap with next plot
else:
    st.warning("State or AQI column not found for visualization")

# AQI Range Distribution
if 'AQI_Range' in df_clean.columns:
    st.write("AQI Range Distribution")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='AQI_Range', data=df_clean, 
                 order=["Good", "Moderate", "Poor", "Unhealthy", "Very Unhealthy", "Hazardous"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
else:
    st.warning("AQI_Range column not found for visualization")

# Model Training
st.subheader("Model Training")

# Check if required columns exist
required_cols = ['SOi', 'Noi', 'Rpi', 'SPMi', 'AQI']
if all(col in df_clean.columns for col in required_cols):
    # Regression models for AQI prediction
    X = df_clean[['SOi', 'Noi', 'Rpi', 'SPMi']]
    Y = df_clean['AQI']

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=70)

    # Model selection
    model_option = st.selectbox("Select a regression model", 
                              ["Linear Regression", "Decision Tree", "Random Forest"])

    if model_option == "Linear Regression":
        model = LinearRegression()
    elif model_option == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    model.fit(X_train, Y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Evaluation
    st.write("Model Evaluation")
    st.write(f"RMSE on Training Data: {np.sqrt(mean_squared_error(Y_train, train_pred)):.2f}")
    st.write(f"RMSE on Test Data: {np.sqrt(mean_squared_error(Y_test, test_pred)):.2f}")
    st.write(f"R-squared on Training Data: {r2_score(Y_train, train_pred):.2f}")
    st.write(f"R-squared on Test Data: {r2_score(Y_test, test_pred):.2f}")

    # Classification for AQI Range
    st.subheader("AQI Range Classification")

    # Select features and target
    X2 = df_clean[['SOi', 'Noi', 'Rpi', 'SPMi']]
    Y2 = df_clean['AQI_Range']

    # Split data
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.33, random_state=70)

    # Model selection
    classifier_option = st.selectbox("Select a classifier", 
                                   ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"])

    if classifier_option == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    elif classifier_option == "Decision Tree":
        classifier = DecisionTreeClassifier(random_state=42)
    elif classifier_option == "Random Forest":
        classifier = RandomForestClassifier(random_state=42)
    else:
        classifier = KNeighborsClassifier()

    classifier.fit(X_train2, Y_train2)

    # Predictions
    train_preds2 = classifier.predict(X_train2)
    test_preds2 = classifier.predict(X_test2)

 
    # Evaluation
    st.write("Classifier Evaluation")
    st.write(f"Accuracy on Training Data: {accuracy_score(Y_train2, train_preds2):.2f}")
    st.write(f"Accuracy on Test Data: {accuracy_score(Y_test2, test_preds2):.2f}")
    
    # Add feature importance for tree-based models
    if classifier_option in ["Decision Tree", "Random Forest"]:
        st.write("Feature Importance:")
        feature_importance = pd.DataFrame({
            'Feature': X2.columns,
            'Importance': classifier.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.write(feature_importance)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        st.pyplot(plt)
        plt.clf()
else:
    st.error("Required columns for modeling not found in the dataset. Please check your data.")



st.title("Air Quality Checker")
st.markdown("Enter a city name to fetch real-time air pollutant levels (SO₂, NO₂, RSPM, SPM).")

# Input city name
city_name = st.text_input("City Name", placeholder="e.g., Delhi, London, Tokyo")

# OpenWeatherMap API key (replace with your own)
API_KEY = "414906df50705deda141ec103d93fdca"  # Note: This key may not work; use your own.

def get_coordinates(city_name, api_key):
    """Fetch latitude and longitude for a city using OpenWeather Geocoding API."""
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
    try:
        response = requests.get(geo_url)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None, None
        lat = data[0]["lat"]
        lon = data[0]["lon"]
        return lat, lon
    except requests.exceptions.RequestException:
        return None, None

def get_air_quality(lat, lon, api_key):
    """Fetch pollutant levels (SO2, NO2, PM2.5, PM10) for given coordinates."""
    if lat is None or lon is None:
        return None
    air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(air_url)
        response.raise_for_status()
        data = response.json()
        pollutants = data["list"][0]["components"]
        return {
            "SO₂": pollutants["so2"],     # Sulfur Dioxide (μg/m³)
            "NO₂": pollutants["no2"],     # Nitrogen Dioxide (μg/m³)
            "PM10": pollutants["pm10"], # Fine Particulate Matter (μg/m³)
            "PM2.5": pollutants["pm2_5"]    # Coarse Particulate Matter (μg/m³)
        }
    except (requests.exceptions.RequestException, KeyError):
        return None

# Fetch and display data when the user clicks the button
if st.button("Check Air Quality"):
    if not city_name:
        st.warning("Please enter a city name.")
    else:
        with st.spinner("Fetching data..."):
            lat, lon = get_coordinates(city_name, API_KEY)
            if lat is None or lon is None:
                st.error("Could not find the city. Please check the name and try again.")
            else:
                pollutants = get_air_quality(lat, lon, API_KEY)
                if pollutants:
                    st.success(f"📍 **City:** {city_name.title()}")
                    st.markdown("---")
                    st.subheader("Pollutant Levels (μg/m³)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("SO₂ (Sulfur Dioxide)", pollutants["SO₂"])
                        st.metric("NO₂ (Nitrogen Dioxide)", pollutants["NO₂"])
                    with col2:
                        st.metric("RSPM (Respirable Suspended Particulate Matter)", pollutants["PM10"])
                        st.metric("SPM (Suspended Particulate Matter)", pollutants["PM2.5"])
                    st.markdown("---")
                    st.info("ℹ️ Data provided by OpenWeatherMap API")
                else:
                    st.error("Failed to fetch air quality data. Please try again later.")




st.subheader("AQI Prediction from User Input")

# Create input fields for pollutants
col1, col2 = st.columns(2)

with col1:
    so2_input = st.number_input("SO₂ level (μg/m³)", min_value=0.0, value=20.0, step=1.0)
    no2_input = st.number_input("NO₂ level (μg/m³)", min_value=0.0, value=30.0, step=1.0)

with col2:
    rspm_input = st.number_input("RSPM level (μg/m³)", min_value=0.0, value=50.0, step=1.0)
    spm_input = st.number_input("SPM level (μg/m³)", min_value=0.0, value=60.0, step=1.0)

# Calculate sub-indices from user input
def calculate_sub_indices(so2, no2, rspm, spm):
    # SO2 sub-index
    if so2 <= 40:
        soi = so2 * (50 / 40)
    elif 40 < so2 <= 80:
        soi = 50 + (so2 - 40) * (50 / 40)
    elif 80 < so2 <= 380:
        soi = 100 + (so2 - 80) * (100 / 300)
    elif 380 < so2 <= 800:
        soi = 200 + (so2 - 380) * (100 / 420)
    elif 800 < so2 <= 1600:
        soi = 300 + (so2 - 800) * (100 / 800)
    else:
        soi = 400 + (so2 - 1600) * (100 / 800)
    
    # NO2 sub-index
    if no2 <= 40:
        noi = no2 * 50 / 40
    elif 40 < no2 <= 80:
        noi = 50 + (no2 - 40) * (50 / 40)
    elif 80 < no2 <= 180:
        noi = 100 + (no2 - 80) * (100 / 100)
    elif 180 < no2 <= 280:
        noi = 200 + (no2 - 180) * (100 / 100)
    elif 280 < no2 <= 400:
        noi = 300 + (no2 - 280) * (100 / 120)
    else:
        noi = 400 + (no2 - 400) * (100 / 120)
    
    # RSPM sub-index
    if rspm <= 30:
        rpi = rspm * 50 / 30
    elif 30 < rspm <= 60:
        rpi = 50 + (rspm - 30) * 50 / 30
    elif 60 < rspm <= 90:
        rpi = 100 + (rspm - 60) * 100 / 30
    elif 90 < rspm <= 120:
        rpi = 200 + (rspm - 90) * 100 / 30
    elif 120 < rspm <= 250:
        rpi = 300 + (rspm - 120) * (100 / 130)
    else:
        rpi = 400 + (rspm - 250) * (100 / 130)
    
    # SPM sub-index
    if spm <= 50:
        spmi = spm * 50 / 50
    elif 50 < spm <= 100:
        spmi = 50 + (spm - 50) * (50 / 50)
    elif 100 < spm <= 250:
        spmi = 100 + (spm - 100) * (100 / 150)
    elif 250 < spm <= 350:
        spmi = 200 + (spm - 250) * (100 / 100)
    elif 350 < spm <= 430:
        spmi = 300 + (spm - 350) * (100 / 80)
    else:
        spmi = 400 + (spm - 430) * (100 / 430)
    
    return soi, noi, rpi, spmi

# Prediction button
if st.button("Predict AQI"):
    # Calculate sub-indices
    soi, noi, rpi, spmi = calculate_sub_indices(so2_input, no2_input, rspm_input, spm_input)
    
    # Calculate AQI (manual calculation)
    manual_aqi = max(soi, noi, rpi, spmi)
    
    # Determine AQI category
    if manual_aqi <= 50:
        category = "Good"
    elif 50 < manual_aqi <= 100:
        category = "Moderate"
    elif 100 < manual_aqi <= 200:
        category = "Poor"
    elif 200 < manual_aqi <= 300:
        category = "Unhealthy"
    elif 300 < manual_aqi <= 400:
        category = "Very Unhealthy"
    else:
        category = "Hazardous"
    
    # Model prediction (if models were trained)
    if 'model' in locals() and 'classifier' in locals():
        # Create input array for model
        input_data = [[soi, noi, rpi, spmi]]
        
        # Regression prediction
        predicted_aqi = model.predict(input_data)[0]
        
        # Classification prediction
        predicted_category = classifier.predict(input_data)[0]
    else:
        predicted_aqi = "N/A (models not trained)"
        predicted_category = "N/A (models not trained)"
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Air Quality Index", f"{manual_aqi:.1f}", category)
