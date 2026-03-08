# scripts/data_preprocessing/merge_with_traffic.py

import os
import sys
import time
import logging
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.neighbors import KDTree

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils.logging_setup import setup_logging
from scripts.config import MERGED_PW_FILE, TRAFFIC_DATA_FILE, FINAL_DATASET_FILE_1


def merge_with_traffic():

    log_file_path = 'logs/data_processing/merge_with_traffic.log'
    setup_logging(log_file_path)

    logging.info("Starting merge_with_traffic process")

    merged_pw_file = MERGED_PW_FILE
    traffic_file = TRAFFIC_DATA_FILE
    final_file = FINAL_DATASET_FILE_1

    chunksize = 10000
    processed_rows = 0

# scaling factor to balance time and space
    TIME_SCALE = 1 / 3600  # convert seconds → hours

    if os.path.exists(final_file):
        os.remove(final_file)
        logging.info(f"Removed existing output file: {final_file}")

    # -----------------------------
    # Load traffic data
    # -----------------------------
    try:
        traffic = pd.read_csv(traffic_file)

        traffic['gps_timestamp'] = pd.to_datetime(traffic['gps_timestamp'])

        traffic['time_numeric'] = (
            traffic['gps_timestamp'].astype('int64') // 10**9
        )

        traffic_coords = np.column_stack([
            traffic['latitude'].values,
            traffic['longitude'].values,
            traffic['time_numeric'].values * TIME_SCALE
        ])

        tree = KDTree(traffic_coords, metric='euclidean')

        logging.info(f"Traffic data loaded: {traffic.shape}")

    except Exception:
        logging.exception("Error loading traffic data")
        raise

    total_rows = sum(1 for _ in open(merged_pw_file)) - 1

    # -----------------------------
    # Process merged_pw in chunks
    # -----------------------------
    for chunk in pd.read_csv(merged_pw_file, chunksize=chunksize):

        start_time = time.time()

        try:

            chunk['gps_timestamp'] = pd.to_datetime(chunk['gps_timestamp'])

            chunk['time_numeric'] = (
                chunk['gps_timestamp'].astype('int64') // 10**9
            )

            query_points = np.column_stack([
                chunk['latitude'].values,
                chunk['longitude'].values,
                chunk['time_numeric'].values * TIME_SCALE
            ])

            dist, idx = tree.query(query_points, k=1)

            nearest = traffic.iloc[idx.flatten()].reset_index(drop=True)

            # assign traffic value
            chunk["avg_v"] = nearest.iloc[:, 3].values

            chunk.drop(columns=['time_numeric'], inplace=True, errors='ignore')

            processed_rows += len(chunk)

            # write output
            if processed_rows == len(chunk):
                chunk.to_csv(final_file, mode='a', header=True, index=False)
            else:
                chunk.to_csv(final_file, mode='a', header=False, index=False)

            elapsed = time.time() - start_time
            progress = (processed_rows / total_rows) * 100

            logging.info(
                f"Progress: {progress:.2f}% | "
                f"Rows processed: {processed_rows} | "
                f"Chunk time: {elapsed:.2f}s"
            )

        except Exception:
            logging.exception("Error processing chunk")
            continue

    logging.info("merge_with_traffic completed successfully")


if __name__ == "__main__":
    merge_with_traffic()