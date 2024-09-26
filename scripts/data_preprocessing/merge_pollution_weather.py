# scripts/data_preprocessing/merge_datasets.py

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import logging
from scripts.utils.logging_setup import setup_logging
from scripts.config import POLLUTION_DATA_FILE, WEATHER_DATA_FILE, MERGED_PW_FILE, PROCESSED_DATA_DIR
import haversine as hs
import multiprocessing
from functools import partial
import time

def merge_pollution_weather(pollution_file, weather_file, output_file):
    """
    Merges the pollution data with the weather data on 'date_hour'.
    Retains all rows from the pollution data.

    Args:
        pollution_file (str): Path to the processed pollution data file.
        weather_file (str): Path to the processed weather data file.
        output_file (str): Path where the merged data will be saved.
    """
    logging.info("Starting merge of pollution and weather data on date and hour...")

    try:
        pollution_df = pd.read_csv(pollution_file, parse_dates=['gps_timestamp'])
        logging.info(f"Loaded pollution data with shape: {pollution_df.shape}")
    except Exception as e:
        logging.exception(f"Error reading pollution data: {e}")
        raise

    try:
        weather_df = pd.read_csv(weather_file, parse_dates=['gps_timestamp'])
        logging.info(f"Loaded weather data with shape: {weather_df.shape}")
    except Exception as e:
        logging.exception(f"Error reading weather data: {e}")
        raise

    # Extract date and hour from 'gps_timestamp'
    pollution_df['date_hour'] = pollution_df['gps_timestamp'].dt.floor('h')
    weather_df['date_hour'] = weather_df['gps_timestamp'].dt.floor('h')

    # Merge on 'date_hour'
    merged_df = pd.merge(
        pollution_df,
        weather_df,
        on='date_hour',
        how='left',  # Retain all pollution data rows
        suffixes=('', '_weather')  # Handle overlapping column names
    )

    logging.info(f"Merged pollution and weather data on date and hour. Shape after merge: {merged_df.shape}")

    # Optionally, drop unnecessary columns from weather data
    columns_to_drop = ['gps_timestamp_weather', 'date_hour', 'date_hour_weather']
    merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')


    # Save the merged data
    merged_df.to_csv(output_file, index=False)
    logging.info(f"Merged pollution and weather data saved to {output_file}")

    


def main_merge_pollution_weather():
    # Set up logging
    log_file_path = 'logs/data_processing/merge_datasets.log'
    setup_logging(log_file_path)

    # Define file paths
    processed_data_dir = PROCESSED_DATA_DIR
    pollution_file = POLLUTION_DATA_FILE
    weather_file = WEATHER_DATA_FILE

    # Output files
    merged_pw_file = MERGED_PW_FILE

    # Ensure processed data directory exists
    if not os.path.exists(processed_data_dir):
        logging.error(f"Processed data directory does not exist: {processed_data_dir}")
        raise FileNotFoundError(f"Processed data directory does not exist: {processed_data_dir}")

    # Merge pollution and weather data
    merge_pollution_weather(pollution_file, weather_file, merged_pw_file)


if __name__ == "__main__":
    main_merge_pollution_weather()

