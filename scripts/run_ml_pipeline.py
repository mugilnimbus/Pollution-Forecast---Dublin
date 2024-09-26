# scripts/run_pipeline.py

import os
import sys
import logging


# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
print(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from utils.logging_setup import setup_logging

from data_preprocessing.ingest_all_zip_files import main_ingester
from data_preprocessing.merge_pollution_weather import main_merge_pollution_weather
from data_preprocessing.process_airview import main_process_airview
from data_preprocessing.process_traffic import main_process_traffic
from data_preprocessing.process_weather import main_process_weather
from data_preprocessing.merge_with_traffic import main_merge_with_traffic
from modeling.train_model import train_model
from modeling.evaluate_model import evaluate_model
from config import LOGS_DIR

def main():
    # Set up logging
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(LOGS_DIR, 'pipeline.log')
    setup_logging(log_file_path)
    logging.info("Starting ML pipeline...")

    try:
        logging.info("Step 1: Ingesting data from zip files...")
        main_ingester()

        logging.info("Step 2: Processing airview pollution data...")
        main_process_airview()

        logging.info("Step 3: Processing traffic data...")
        main_process_traffic()

        logging.info("Step 4: Processing weather data...")
        main_process_weather()

        logging.info("Step 5: Merging pollution and weather data...")
        main_merge_pollution_weather()

        logging.info("Step 6: Merging with traffic data...")
        main_merge_with_traffic()

        logging.info("Step 7: Training the model...")
        train_model()

        logging.info("Step 8: Evaluating the model...")
        evaluate_model()

        logging.info("ML pipeline completed successfully.")

    except Exception as e:
        logging.exception(f"Pipeline failed due to an error: {e}")

if __name__ == "__main__":
    main()
