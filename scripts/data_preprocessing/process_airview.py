# scripts/data_preprocessing/process_airview.py

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import logging
from scripts.utils.logging_setup import setup_logging
from scripts.config import EXTRACTED_DATA_DIR, PROCESSED_DATA_DIR, POLLUTION_DATA_FILE

def process_airview_data(extracted_dir, output_path):
    """
    Processes the AirView data and saves the cleaned data to output_path.

    Args:
        extracted_dir (str): Path to the extracted data directory.
        output_path (str): Path where the processed data will be saved.
    """
    logging.info("Starting AirView data processing...")

    # Path to the extracted AirView CSV files
    airview_dir = os.path.join(extracted_dir, 'airview')

    # Check if the AirView directory exists
    if not os.path.exists(airview_dir):
        logging.error(f"AirView directory does not exist: {airview_dir}")
        raise FileNotFoundError(f"AirView directory does not exist: {airview_dir}")

    # Find all CSV files in the AirView directory
    csv_files = [os.path.join(airview_dir, f) for f in os.listdir(airview_dir) if f.endswith('.csv')]

    if not csv_files:
        logging.error("No CSV files found in extracted AirView data.")
        raise FileNotFoundError("No CSV files found in extracted AirView data.")

    # Read and concatenate all CSV files
    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            logging.info(f"Loaded {csv_file} with shape: {df.shape}")
            df_list.append(df)
        except Exception as e:
            logging.exception(f"Error reading {csv_file}: {e}")

    if not df_list:
        logging.error("No data frames were loaded. Exiting processing.")
        raise ValueError("No data frames were loaded. Exiting processing.")

    df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined AirView data shape: {df.shape}")

    # Data cleaning and preprocessing

    # Drop unnecessary columns
    columns_to_drop = ["PMch1_perL", "PMch2_perL", "PMch3_perL", "PMch4_perL", "PMch5_perL", "PMch6_perL"]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    logging.debug(f"Dropped columns: {columns_to_drop}")

    # Convert 'gps_timestamp' to datetime
    df['gps_timestamp'] = pd.to_datetime(df['gps_timestamp'], errors='coerce')
    df['gps_timestamp'] = df['gps_timestamp'].dt.tz_localize(None)
    logging.debug("Converted 'gps_timestamp' to datetime and removed timezone.")

    # Drop rows with missing timestamps
    initial_shape = df.shape
    df.dropna(subset=['gps_timestamp'], inplace=True)
    logging.debug(f"Dropped rows with missing 'gps_timestamp'. Rows before: {initial_shape[0]}, after: {df.shape[0]}")

    # Sort data
    df.sort_values(by=["gps_timestamp", "latitude", "longitude"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.debug("Sorted data by 'gps_timestamp', 'latitude', and 'longitude'.")

    logging.info(f"Processed AirView data shape: {df.shape}")

    # Save processed data
    df.to_csv(output_path, index=False)
    logging.info(f"Processed AirView data saved to {output_path}")

def main_process_airview():

    # Run the data processing function
    extracted_data_dir = EXTRACTED_DATA_DIR
    processed_data_dir = PROCESSED_DATA_DIR
    os.makedirs(processed_data_dir, exist_ok=True)
    airview_output = POLLUTION_DATA_FILE

    process_airview_data(extracted_data_dir, airview_output)



if __name__ == "__main__":
    # Set up logging
    log_file_path = 'logs/data_processing/process_airview.log'
    setup_logging(log_file_path)

    main_process_airview()


