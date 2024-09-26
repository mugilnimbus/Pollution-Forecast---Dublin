# scripts/data_preprocessing/process_weather.py

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import logging
from scripts.utils.logging_setup import setup_logging
from scripts.config import EXTRACTED_DATA_DIR, PROCESSED_DATA_DIR, WEATHER_DATA_FILE

def process_weather_data(extracted_dir, output_path):
    """
    Processes the weather data and saves the cleaned data to output_path.

    Args:
        extracted_dir (str): Path to the extracted data directory.
        output_path (str): Path where the processed data will be saved.
    """
    logging.info("Starting weather data processing...")

    # Path to the extracted weather CSV file
    weather_dir = os.path.join(extracted_dir, 'weather')
    weather_file = 'hly532.csv'
    weather_file_path = os.path.join(weather_dir, weather_file)

    # Check if the weather file exists
    if not os.path.exists(weather_file_path):
        logging.error(f"Weather file does not exist: {weather_file_path}")
        raise FileNotFoundError(f"Weather file does not exist: {weather_file_path}")

    # Read the weather data, skipping the first 23 rows
    try:
        weather_df = pd.read_csv(
            weather_file_path,
            skiprows=23,
            engine='python'
        )
        logging.info(f"Loaded weather data from {weather_file_path} with shape: {weather_df.shape}")
    except Exception as e:
        logging.exception(f"Error reading weather file: {e}")
        raise

    # Rename 'date' column to 'gps_timestamp'
    if 'date' not in weather_df.columns:
        logging.error("'date' column not found in weather data.")
        raise KeyError("'date' column not found in weather data.")
    weather_df.rename(columns={"date": "gps_timestamp"}, inplace=True)
    logging.debug("Renamed 'date' column to 'gps_timestamp'.")

    # Convert 'gps_timestamp' to datetime
    weather_df['gps_timestamp'] = pd.to_datetime(
        weather_df['gps_timestamp'],
        format="%d-%b-%Y %H:%M",
        errors='coerce'
    )
    logging.debug("Converted 'gps_timestamp' to datetime.")

    # Extract 'hour' from 'gps_timestamp'
    weather_df['hour'] = weather_df['gps_timestamp'].dt.hour
    logging.debug("Extracted 'hour' from 'gps_timestamp'.")

    # Filter data for hours between 7 and 20 (inclusive)
    initial_shape = weather_df.shape
    weather_df = weather_df[(weather_df['hour'] > 6.0) & (weather_df['hour'] <= 20.0)]
    logging.debug(f"Filtered data for hours between 7 and 20. Rows before: {initial_shape[0]}, after: {weather_df.shape[0]}")


    # Drop unnecessary columns
    columns_to_drop = ["ind", "ind.1", "ind.2", "ind.3", "ind.4", "wetb", "dewpt", "vappr", "msl", "ww", "w", "sun", "vis", "clht", "clamt"]
    weather_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    logging.debug(f"Dropped columns: {columns_to_drop}")

    # Handle missing values
    missing_values_count = weather_df.isna().sum()
    logging.debug(f"Missing values before dropna:\n{missing_values_count}")
    weather_df.dropna(inplace=True)
    logging.debug("Dropped rows with missing values.")

    # Reset index
    weather_df.reset_index(drop=True, inplace=True)

    # Log final data shape
    logging.info(f"Processed weather data shape: {weather_df.shape}")

    # Save the processed weather data
    weather_df.to_csv(output_path, index=False)
    logging.info(f"Processed weather data saved to {output_path}")

def main_process_weather():
    # Set up logging
    log_file_path = 'logs/data_processing/process_weather.log'
    setup_logging(log_file_path)

    # Define the extracted data directory and output file path
    extracted_data_dir = EXTRACTED_DATA_DIR
    processed_data_dir = PROCESSED_DATA_DIR
    os.makedirs(processed_data_dir, exist_ok=True)
    weather_output = WEATHER_DATA_FILE

    # Run the data processing function
    process_weather_data(extracted_data_dir, weather_output)



if __name__ == "__main__":
    main_process_weather()

