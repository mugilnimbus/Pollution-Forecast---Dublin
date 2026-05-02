# scripts/data_preprocessing/ingest_all_zip_files.py

import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import zipfile
import logging
from scripts.utils.logging_setup import setup_logging
from scripts.config import RAW_DATA_DIR,EXTRACTED_DATA_DIR


def extract_zip_files(raw_data_dir: str, extracted_data_dir: str):
    """
    Extracts all zip files from the raw data directory into the extracted data directory,
    maintaining the directory structure.

    Args:
        raw_data_dir (str): Path to the raw data directory containing zip files.
        extracted_data_dir (str): Path to the directory where extracted files will be stored.
    """
    logging.info(f"Starting extraction of zip files from {raw_data_dir}")

    # Check if the raw data directory exists
    if not os.path.exists(raw_data_dir):
        logging.error(f"Raw data directory does not exist: {raw_data_dir}")
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    # Walk through the raw data directory
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.lower().endswith('.zip'):
                # Full path to the zip file
                zip_file_path = os.path.join(root, file)

                # Relative path to maintain directory structure
                relative_path = os.path.relpath(root, raw_data_dir)

                # Destination directory in extracted_data_dir
                destination_dir = os.path.join(extracted_data_dir, relative_path)

                # Ensure destination directory exists
                os.makedirs(destination_dir, exist_ok=True)

                # Extract zip file
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(destination_dir)
                        logging.info(f"Extracted {zip_file_path} to {destination_dir}")
                except zipfile.BadZipFile:
                    logging.error(f"Failed to extract {zip_file_path}: Bad zip file")
                except Exception as e:
                    logging.exception(f"An error occurred while extracting {zip_file_path}: {e}")
            else:
                logging.debug(f"Skipping non-zip file: {os.path.join(root, file)}")

    logging.info("All zip files have been extracted.")

def main_ingester():

    # Define your raw data directory and extracted data directory
    raw_data_dir = RAW_DATA_DIR
    extracted_data_dir = EXTRACTED_DATA_DIR

    # Ensure extracted data directory exists
    os.makedirs(extracted_data_dir, exist_ok=True)

    # Run the extraction function
    extract_zip_files(raw_data_dir, extracted_data_dir)


if __name__ == "__main__":
    # Set up logging
    log_file_path = 'logs/data_ingestion/ingest_all_zip_files.log'
    setup_logging(log_file_path)
    
    main_ingester()
