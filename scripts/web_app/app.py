# app.py

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils.logging_setup import setup_logging
from scripts.config import TRAINED_MODEL_FILE,SCALER_FILE

import logging
from flask import Flask, render_template, request, redirect, url_for, session
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import joblib
import time
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Users dictionary for authentication (not persistent)
users = {}

# Mapping for average traffic volume
avg_tv = [60, 220, 50, 120, 60, 150, 50, 100, 170, 80, 130, 90, 200, 100]
hours_t = list(range(7, 21))
value_dict = dict(zip(hours_t[:len(avg_tv)], avg_tv))

hour_l=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

def fetch_weather_data(latitude, longitude):
    url = f"http://metwdb-openaccess.ichec.ie/metno-wdb2ts/locationforecast?lat={latitude};long={longitude}"
    response = requests.get(url)
    #print(response)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        product = root.find('product')
        data = []

        for time_element in product.findall('time'):
            from_time = time_element.attrib['from']
            to_time = time_element.attrib['to']

            location = time_element.find('location')

            row_data = {
                'from_time': from_time,
                'to_time': to_time
            }

            # Extract weather elements
            for element_name in ['temperature', 'humidity', 'windSpeed','windDirection', 'precipitation']:
                element = location.find(element_name)
                if element is not None:
                    value_key = 'value' if element_name != 'windSpeed' else 'mps'
                    if element_name == 'windDirection':
                        value_key = 'deg'
                    row_data[element_name.lower()] = float(element.get(value_key))

            data.append(row_data)

        df = pd.DataFrame(data)
        print(df)
        # Ensure all expected columns are present
        df = df[['from_time', 'temperature', 'humidity', 'windspeed','winddirection', 'precipitation']]
        print(df.head())
        df = df.rename(columns={df.columns[0]: "from_time"})
        time.sleep(5)
        df['precipitation'] = df['precipitation'].shift(-1)
        df.dropna(inplace=True)
        #df.drop(["to_time"],axis=1,inplace=True)
        print(df.columns)
        df = df.iloc[:72]
        df['from_time'] = pd.to_datetime(df['from_time'], format="%Y-%m-%dT%H:%M:%SZ")
        df['dayofweek'] = df['from_time'].dt.dayofweek
        df['month'] = df['from_time'].dt.month
        df['hour'] = df['from_time'].dt.hour
        df = df[(df['hour'] >= 7) & (df['hour'] <= 20)]
        df["longitude"] = float(longitude)
        df["latitude"] = float(latitude)
        df['avg_v'] = df['hour'].map(value_dict)

        df.reset_index(drop=True, inplace=True)

        return df
    else:
        logging.error(f"Failed to fetch weather data. Status code: {response.status_code}")
        return None

def predicter(w_df):
    w_df = w_df[['precipitation', 'temperature', 'humidity', 'windspeed','winddirection','hour','avg_v','month', 'dayofweek']]
    x_input = w_df.values

    # Load model and scaler
    model_path = TRAINED_MODEL_FILE
    scaler_path = SCALER_FILE
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logging.error("Model or scaler file not found.")
        return None

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Transform input data
    x_vals = scaler.transform(x_input)

    # Make predictions
    predictions = model.predict(x_vals).flatten()

    # Plot predictions
    plotter(predictions)
    return predictions

def plotter(predictions):
    chunks = [14, 28, 42]
    start = 0

    # Set up the figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Set global figure properties
    fig.patch.set_facecolor('black')

    for i, end in enumerate(chunks):
        # Plot data on each subplot
        axs[i].plot(hour_l[start:end], predictions[start:end], color='white')
        axs[i].set_facecolor('black')
        axs[i].tick_params(axis='both', colors='white')
        start = end
    
    plt.title('Predicted PM2.5 Levels')
    plt.xlabel('Time (Hours)')
    plt.ylabel('PM2.5 Level')
    
    plt.tight_layout()
    plt.savefig('scripts/web_app/static/Fig1.png')
    plt.close()

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate user
        if users.get(username) == password:
            session['username'] = username
            return redirect(url_for('homepage'))
        else:
            # Invalid credentials
            return "Invalid username or password", 401

    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users:
            return "Username already exists. Please choose another one.", 400

        # Add user to the users dictionary
        users[username] = password

        return redirect(url_for('signin'))

    return render_template('signup.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        coordinates = request.form.get('coordinates')
        date = request.form.get('date')
        if not coordinates or not date:
            return "Coordinates and date are required.", 400

        latitude, longitude = map(float, coordinates.split(","))
        
        # Fetch weather data
        weather_df = fetch_weather_data(latitude, longitude)
        if weather_df is None or weather_df.empty:
            return "Failed to fetch weather data. Please try again.", 500

        # Run predictions
        predictions = predicter(weather_df)
        if predictions is None:
            return "Prediction failed due to an error. Please try again later.", 500

        return render_template('home.html', plot_url='Fig1.png')
    except Exception as e:
        logging.exception(f"Error in /predict route: {e}")
        return "An error occurred during prediction. Please try again.", 500

if __name__ == "__main__":
    # Set up logging
    logs_dir = os.path.join('logs', 'web_app')
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, 'app.log')
    setup_logging(log_file_path)


    app.run(debug=True)
