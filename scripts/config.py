# scripts/config.py

import os

# Directories
DATA_DIR = os.path.join('data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTRACTED_DATA_DIR = os.path.join(DATA_DIR, 'extracted')
MODELS_DIR = os.path.join('models')
EVALUATION_DIR = os.path.join( MODELS_DIR,'evaluation')
LOGS_DIR = os.path.join('logs')

# Files
POLLUTION_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'pollution_data.csv')
WEATHER_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'weather_data.csv')
TRAFFIC_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'traffic_data.csv')
MERGED_PW_FILE = os.path.join(PROCESSED_DATA_DIR, 'pollution_weather_data.csv')
FINAL_DATASET_FILE = os.path.join(PROCESSED_DATA_DIR, 'final_dataset.csv')

TRAINED_MODEL_FILE = os.path.join(MODELS_DIR, 'trained_model.h5')
SCALER_FILE = os.path.join(MODELS_DIR, 'scaler.pkl')
