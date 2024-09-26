

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import logging
from scripts.utils.logging_setup import setup_logging
import pandas as pd
import datetime as dt
import haversine as hs
import time
import os


# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))


# Set up logging
log_file_path = f'logs/data_processing/batch_logs/{os.path.basename(script_dir)}.log'
setup_logging(log_file_path)


try:
    t1 = dt.datetime.strptime("2021-05-06 06:00:00", "%Y-%m-%d %H:%M:%S")
    t2 = dt.datetime.strptime("2021-05-06 08:00:00", "%Y-%m-%d %H:%M:%S")
    logging.info(f"Script started with t1: {t1}, t2: {t2}")
except Exception as e:
    logging.exception("Error parsing initial timestamps")
    raise

days = 456 * 12
c_rows = 0


# Construct file paths
merged_pw_file = os.path.join(script_dir, "merged_pw_batch.csv")
traffic_file = os.path.join(script_dir, "traffic_batch.csv")
final_file=os.path.join(script_dir, "final.csv")

# Check if the file exists
logging.info(f"Checking for final output file exists from {final_file}")
if os.path.exists(final_file):
    # Delete the file
    os.remove(final_file)
    logging.info(f"Successfully deleted existing output file from {final_file}")
    print(f"{final_file} has been deleted.")
else:
    logging.info(f"{final_file} does not exist.")
    print(f"{final_file} does not exist.")

t_rows = len(pd.read_csv(merged_pw_file))  #5030143  # Total number of rows (for progress calculation)

# Read the entire traffic_file once to avoid reading it multiple times
try:
    df8 = pd.read_csv(traffic_file)
    df8['gps_timestamp'] = pd.to_datetime(df8['gps_timestamp'], format="%Y-%m-%d %H:%M:%S")
    logging.info(f"Successfully read traffic data from {traffic_file}")
except Exception as e:
    logging.exception("Error reading traffic_file")
    raise  # Stop execution if traffic_file can't be read

# Process merged_pw_file in chunks
for P in pd.read_csv(merged_pw_file, chunksize=5000):
    start_time = time.time()
    try:
        # Convert gps_timestamp to datetime
        P['gps_timestamp'] = pd.to_datetime(P['gps_timestamp'], format='%Y-%m-%d %H:%M:%S')
        t1 = P.iloc[0]['gps_timestamp']
        t2 = P.iloc[-1]['gps_timestamp']
        c_rows += len(P)

        # Filter df8 for the current time window
        df10 = df8[(df8['gps_timestamp'] >= t1) & (df8['gps_timestamp'] <= t2)]

        if not P.empty and not df10.empty:
            avg = []
            for _, r in P.iterrows():
                df10_copy = df10.copy()
                df10_copy['abs_diff_A'] = abs(df10_copy['gps_timestamp'] - r['gps_timestamp'])
                cords = [(r['latitude'], r['longitude'])] * len(df10_copy)
                coords = list(zip(df10_copy['latitude'], df10_copy['longitude']))
                df10_copy['abs_diff_B'] = hs.haversine_vector(cords, coords, unit=hs.Unit.METERS)
                sorted_df = df10_copy.sort_values(by=['abs_diff_A', "abs_diff_B"])
                avg_value = sorted_df.iloc[0, 3]  # Adjust the column index or use column name
                avg.append(avg_value)

            P["avg_v"] = avg

            # Write to CSV
            if c_rows == 5000:
                P.to_csv(final_file, mode='a', header=True, index=False)
                logging.info(f"Created final.csv and wrote first chunk of {len(P)} rows.")
            else:
                P.to_csv(final_file, mode='a', header=False, index=False)
                logging.info(f"Appended chunk of {len(P)} rows to final.csv.")
        else:
            logging.warning(f"No data to process for chunk starting at row {c_rows - len(P)}")

    except Exception as e:
        logging.exception(f"Error processing chunk starting at row {c_rows - len(P)}")
        continue  # Skip to the next chunk

    elapsed_time = time.time() - start_time
    progress_percent = (c_rows / t_rows) * 100
    logging.info(f"Progress: {progress_percent:.2f}% | Completed rows: {c_rows} | Time taken for this chunk: {elapsed_time:.2f} seconds")

logging.info(f"Processing {script_dir[57:]} completed successfully.")


