# scripts/modeling/evaluate_model.py

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from joblib import load
from scripts.utils.logging_setup import setup_logging
from scripts.config import MODELS_DIR, TRAINED_MODEL_FILE, SCALER_FILE, FINAL_DATASET_FILE, EVALUATION_DIR

def load_and_prepare_data(data_path):
    """
    Loads and preprocesses the data for evaluation.

    Args:
        data_path (str): Path to the data file.

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
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

        X = df.drop(columns=['PM25_ugm3'])
        y = df['PM25_ugm3']
        logging.info(f"Features and target variable prepared. Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    except Exception as e:
        logging.exception(f"Error preprocessing data: {e}")
        raise

def evaluate_model():
    """
    Loads the model and scaler, evaluates the model on the dataset, and saves evaluation results.
    """
    logging.info("Starting model evaluation...")

    # Define paths
    data_path = FINAL_DATASET_FILE
    models_dir = MODELS_DIR
    evaluation_dir = EVALUATION_DIR
    os.makedirs(evaluation_dir, exist_ok=True)

    # Check if data file exists
    if not os.path.exists(data_path):
        logging.error(f"Data file does not exist: {data_path}")
        raise FileNotFoundError(f"Data file does not exist: {data_path}")

    # Check if model and scaler exist
    scaler_path = SCALER_FILE
    model_path = TRAINED_MODEL_FILE

    if not os.path.exists(scaler_path):
        logging.error(f"Scaler file does not exist: {scaler_path}")
        raise FileNotFoundError(f"Scaler file does not exist: {scaler_path}")

    if not os.path.exists(model_path):
        logging.error(f"Model file does not exist: {model_path}")
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    try:
        # Load and prepare data
        X, y = load_and_prepare_data(data_path)

        # Load scaler and model
        logging.info("Loading scaler and model...")
        scaler = load(scaler_path)
        model = load_model(model_path)
        logging.info("Scaler and model loaded successfully.")

        # Scale features
        X_scaled = scaler.transform(X)
        logging.info("Features scaled using loaded scaler.")

        # Predict
        logging.info("Making predictions on the dataset...")
        y_pred = model.predict(X_scaled).flatten()
        logging.info("Predictions completed.")

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        logging.info(f"Evaluation Metrics - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")

        # Save metrics to a file
        metrics_path = os.path.join(evaluation_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Mean Squared Error: {mse:.4f}\n")
            f.write(f"R-squared: {r2:.4f}\n")
        logging.info(f"Evaluation metrics saved to {metrics_path}")

        # Plot results
        logging.info("Creating evaluation plot...")
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(y)), y, label='Actual', alpha=0.5)
        plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.5)
        plt.legend()
        plt.title('Actual vs Predicted PM2.5 Levels')
        plt.xlabel('Sample Index')
        plt.ylabel('PM2.5 Level')
        plt.tight_layout()
        plot_path = os.path.join(evaluation_dir, 'evaluation_plot.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Evaluation plot saved to {plot_path}")

    except Exception as e:
        logging.exception(f"Error during model evaluation: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logs_dir = os.path.join('logs', 'modeling')
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, 'evaluate_model.log')
    setup_logging(log_file_path)

    # Run the evaluation function
    evaluate_model()

