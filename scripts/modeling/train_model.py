# scripts/modeling/train_model.py

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from joblib import dump
from scripts.utils.logging_setup import setup_logging
from scripts.config import FINAL_DATASET_FILE, MODELS_DIR, TRAINED_MODEL_FILE, SCALER_FILE



def load_data(data_path):
    """
    Loads the data from the specified path and preprocesses it.

    Args:
        data_path (str): Path to the data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded from {data_path} with shape {df.shape}")
    except Exception as e:
        logging.exception(f"Error loading data from {data_path}: {e}")
        raise

    try:
        # Handle datetime features
        df['gps_timestamp'] = pd.to_datetime(df['gps_timestamp'])
        df['month'] = df['gps_timestamp'].dt.month
        df['dayofweek'] = df['gps_timestamp'].dt.dayofweek
        df['hour'] = df['gps_timestamp'].dt.hour
        df.drop(columns=['gps_timestamp'], inplace=True)
        logging.debug("Extracted datetime features and dropped 'gps_timestamp' column.")

        # Drop missing values
        initial_shape = df.shape
        df.dropna(inplace=True)
        logging.info(f"Dropped missing values. Rows before: {initial_shape[0]}, after: {df.shape[0]}")

        return df
    except Exception as e:
        logging.exception(f"Error preprocessing data: {e}")
        raise

def build_model(input_dim):
    """
    Builds and compiles the neural network model.

    Args:
        input_dim (int): Number of input features.

    Returns:
        keras.Model: Compiled Keras model.
    """
    try:
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)  # Regression output
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        logging.info("Model built and compiled.")
        return model
    except Exception as e:
        logging.exception(f"Error building model: {e}")
        raise

def train_model():
    """
    Loads data, trains the model, and saves the trained model and scaler.
    """
    logging.info("Starting model training...")

    # Define paths
    data_path = FINAL_DATASET_FILE
    models_dir = MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    # Check if data file exists
    if not os.path.exists(data_path):
        logging.error(f"Data file does not exist: {data_path}")
        raise FileNotFoundError(f"Data file does not exist: {data_path}")

    try:
        # Load and prepare data
        df = load_data(data_path)
        print(df.head())
        X = df.drop(columns=['PM25_ugm3','latitude','longitude','NO_ugm3','NO2_ugm3','O3_ugm3','CO_mgm3','CO2_mgm3'])
        y = df['PM25_ugm3']
        logging.info(f"Data prepared for training. Features shape: {X.shape}, Target shape: {y.shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        logging.info(f"Data split into training and testing sets. Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Features scaled using StandardScaler.")

        # Build and train model
        model = build_model(input_dim=X_train_scaled.shape[1])
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50, batch_size=32,
            validation_split=0.1,
            callbacks=[]  # You can add callbacks here
        )
        logging.info("Model training completed.")

        # Evaluate the model
        test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
        logging.info(f"Model evaluation on test set - Loss: {test_loss}, MAE: {test_mae}")

        # Save model and scaler
        model_save_path = TRAINED_MODEL_FILE
        model.save(model_save_path)
        scaler_save_path = SCALER_FILE
        dump(scaler, scaler_save_path)
        logging.info(f"Model saved to {model_save_path}")
        logging.info(f"Scaler saved to {scaler_save_path}")

    except Exception as e:
        logging.exception(f"Error during model training: {e}")
        raise



if __name__ == "__main__":

    # Set up logging
    logs_dir = os.path.join('logs', 'modeling')
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, 'train_model.log')
    setup_logging(log_file_path)

    # Run the training function
    train_model()
