# scripts/data_preprocessing/process_traffic.py

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import glob
import logging
from scripts.utils.logging_setup import setup_logging
from scripts.config import EXTRACTED_DATA_DIR, PROCESSED_DATA_DIR, TRAFFIC_DATA_FILE

def process_traffic_data(extracted_dir, output_path):
    """
    Processes the traffic data and saves the cleaned data to output_path.

    Args:
        extracted_dir (str): Path to the extracted data directory.
        output_path (str): Path where the processed data will be saved.
    """
    logging.info("Starting traffic data processing...")

    # Path to the extracted traffic CSV files
    traffic_dir = os.path.join(extracted_dir, 'traffic')

    # Check if the traffic directory exists
    if not os.path.exists(traffic_dir):
        logging.error(f"Traffic directory does not exist: {traffic_dir}")
        raise FileNotFoundError(f"Traffic directory does not exist: {traffic_dir}")

    # Find all SCATS*.csv files in the traffic directory
    scats_files_pattern = os.path.join(traffic_dir, 'SCATS*.csv')
    scats_files = glob.glob(scats_files_pattern)

    if not scats_files:
        logging.error("No SCATS CSV files found in extracted traffic data.")
        raise FileNotFoundError("No SCATS CSV files found in extracted traffic data.")

    # Read and concatenate all SCATS CSV files
    df_list = []
    for scats_file in scats_files:
        try:
            df = pd.read_csv(scats_file, usecols=["End_Time", "Site", "Avg_Volume"])
            logging.info(f"Loaded {scats_file} with shape: {df.shape}")
            df_list.append(df)
        except Exception as e:
            logging.exception(f"Error reading {scats_file}: {e}")

    if not df_list:
        logging.error("No data frames were loaded from SCATS files. Exiting processing.")
        raise ValueError("No data frames were loaded from SCATS files. Exiting processing.")

    traffic_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined traffic data shape: {traffic_df.shape}")

    # Convert 'End_Time' to datetime
    traffic_df['End_Time'] = pd.to_datetime(traffic_df['End_Time'], format="%Y%m%d%H%M%S", errors='coerce')
    logging.debug("Converted 'End_Time' to datetime.")

    # Remove rows with missing 'End_Time'
    initial_shape = traffic_df.shape
    traffic_df.dropna(subset=['End_Time'], inplace=True)
    logging.debug(f"Dropped rows with missing 'End_Time'. Rows before: {initial_shape[0]}, after: {traffic_df.shape[0]}")

    # Extract hour from 'End_Time'
    traffic_df['hour'] = traffic_df['End_Time'].dt.hour
    logging.debug("Extracted 'hour' from 'End_Time'.")

    # Load traffic signals data
    signals_file = os.path.join(traffic_dir, 'dcc_traffic_signals_20221130.csv')
    if not os.path.exists(signals_file):
        logging.error(f"Signals file not found: {signals_file}")
        raise FileNotFoundError(f"Signals file not found: {signals_file}")

    try:
        signals_df = pd.read_csv(signals_file)
        logging.info(f"Loaded signals data from {signals_file} with shape: {signals_df.shape}")
    except Exception as e:
        logging.exception(f"Error reading signals file: {e}")
        raise

    # Filter and rename columns in signals data
    signals_df = signals_df[signals_df['Site_Type'] == "SCATS Site"]
    signals_df.rename(columns={"SiteID": "Site"}, inplace=True)
    signals_df = signals_df[['Site', 'Lat', 'Long']]
    logging.debug("Filtered and renamed columns in signals data.")

    # Merge traffic data with signals data
    merged_df = pd.merge(traffic_df, signals_df, on='Site', how='inner')
    logging.info(f"Merged traffic data with signals data. Shape after merge: {merged_df.shape}")

    # Drop 'Site' column
    merged_df.drop(columns=['Site'], inplace=True)
    logging.debug("Dropped 'Site' column after merging.")

    # Group and aggregate data
    grouped_df = merged_df.groupby(['End_Time', 'Lat', 'Long']).mean().reset_index()
    logging.info(f"Grouped and aggregated data. Shape after grouping: {grouped_df.shape}")

    # Rename columns and filter hours
    grouped_df.rename(columns={"End_Time": "gps_timestamp", "Lat": "latitude", "Long": "longitude"}, inplace=True)
    filtered_df = grouped_df[(grouped_df['hour'] > 6) & (grouped_df['hour'] <= 20)].copy()
    logging.debug(f"Filtered data for hours between 7 and 20. Shape after filtering: {filtered_df.shape}")

    # Sort data and reset index
    filtered_df.sort_values(by=["gps_timestamp", "latitude", "longitude"], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    logging.debug("Sorted data and reset index.")

    logging.info(f"Processed traffic data shape: {filtered_df.shape}")

    # Save processed data
    filtered_df.to_csv(output_path, index=False)
    logging.info(f"Processed traffic data saved to {output_path}")


def main_process_traffic():
    # Set up logging
    log_file_path = 'logs/data_processing/process_traffic.log'
    setup_logging(log_file_path)

    # Run the data processing function
    extracted_data_dir = EXTRACTED_DATA_DIR
    processed_data_dir = PROCESSED_DATA_DIR
    os.makedirs(processed_data_dir, exist_ok=True)
    traffic_output = TRAFFIC_DATA_FILE

    process_traffic_data(extracted_data_dir, traffic_output)


if __name__ == "__main__":
    main_process_traffic()
