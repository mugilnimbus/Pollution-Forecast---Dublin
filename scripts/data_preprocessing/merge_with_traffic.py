# scripts/data_preprocessing/merge_with_traffic.py

import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import logging
from scripts.utils.logging_setup import setup_logging
from scripts.config import PROCESSED_DATA_DIR, MERGED_PW_FILE, TRAFFIC_DATA_FILE, FINAL_DATASET_FILE
import time

def split_batch(num_batches, merged_pw_file, traffic_file):
    """
    Splits the data into batches and prepares individual scripts for processing.
    """
    logging.info("Starting data splitting into batches...")

    # Read the merged pollution and weather data
    try:
        merged_df = pd.read_csv(merged_pw_file, parse_dates=['gps_timestamp'])
        logging.info(f"Loaded merged pollution and weather data with shape: {merged_df.shape}")
    except Exception as e:
        logging.exception(f"Error reading merged pollution and weather data: {e}")
        raise

    # Read the traffic data
    try:
        traffic_df = pd.read_csv(traffic_file, parse_dates=['gps_timestamp'])
        logging.info(f"Loaded traffic data with shape: {traffic_df.shape}")
    except Exception as e:
        logging.exception(f"Error reading traffic data: {e}")
        raise

    # Convert 'gps_timestamp' to datetime if not already done
    merged_df['gps_timestamp'] = pd.to_datetime(merged_df['gps_timestamp'])
    traffic_df['gps_timestamp'] = pd.to_datetime(traffic_df['gps_timestamp'])

    batch_size = int(len(merged_df) / num_batches)
    logging.info(f"Batch size: {batch_size}")

    scripts = []

    for i in range(num_batches):
        batch_dir = f"batch/batch_{i+1}"
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)

        # Save batch of merged data
        batch_merged_file = os.path.join(batch_dir, "merged_pw_batch.csv")
        batch_df = merged_df.iloc[i * batch_size:(i + 1) * batch_size]
        batch_df.to_csv(batch_merged_file, index=False)

        # Filter traffic data for the batch time range
        t1 = batch_df['gps_timestamp'].min()
        t2 = batch_df['gps_timestamp'].max()
        batch_traffic_df = traffic_df[(traffic_df['gps_timestamp'] >= t1) & (traffic_df['gps_timestamp'] <= t2)]
        batch_traffic_file = os.path.join(batch_dir, "traffic_batch.csv")
        batch_traffic_df.to_csv(batch_traffic_file, index=False)

        # Create merge script for the batch
        merge_script_content = create_merge_script()
        merge_script_path = os.path.join(batch_dir, "merge_batch.py")
        with open(merge_script_path, 'w') as file:
            file.write(merge_script_content)

        scripts.append(merge_script_path)

        logging.info(f"Prepared batch {i+1} in {batch_dir}")

    return scripts

def create_merge_script():
    """
    Returns the content of the merge script to be used for each batch.
    """
    script_content = """

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


"""
    return script_content

def run_script(script_path):
    print(f"Running script: {script_path}")

    # Record start time
    start_time = datetime.now()
    print(f"Start time: {start_time}")

    try:
        # Run the Python script using subprocess
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")
    
    # Record end time
    end_time = datetime.now()
    print(f"End time: {end_time}")
    time_taken = end_time - start_time
    print(f"Time taken to run {script_path}: {time_taken}\n")

    return script_path, time_taken

def run_scripts_in_folders_concurrently(scripts, max_workers=20):
    """
    Runs the merge scripts in parallel using multiprocessing.
    """
    logging.info("Starting parallel execution of merge scripts...")
    
    scripts_to_run=scripts

    # Run scripts concurrently using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all the scripts to the executor for parallel execution
        futures = {executor.submit(run_script, script): script for script in scripts_to_run}

        # Gather the results as they are completed
        for future in as_completed(futures):
            script_path, time_taken = future.result()
            print(f"Completed: {script_path} in {time_taken}")
    
    logging.info("All merge scripts have completed.")


def merge_batches(num_batches, final_output_file):
    """
    Merges the result files from each batch into a single final dataset.
    """
    logging.info("Starting merging of batch results...")
    merged_files = []

    for i in range(num_batches):
        batch_dir = f"batch/batch_{i+1}"
        result_file = os.path.join(batch_dir, "final.csv")
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            merged_files.append(df)
            logging.info(f"Loaded result from {result_file}")
        else:
            logging.error(f"Result file not found: {result_file}")

    if merged_files:
        final_df = pd.concat(merged_files, ignore_index=True)
        final_df.to_csv(final_output_file, index=False)
        logging.info(f"Final merged data saved to {final_output_file}")
    else:
        logging.error("No batch result files were found. Final dataset was not created.")
    


    
def main_merge_with_traffic():
    # Set up logging
    log_file_path = 'logs/data_processing/merge_with_traffic.log'
    setup_logging(log_file_path)

    # Define file paths
    processed_data_dir = PROCESSED_DATA_DIR
    merged_pw_file = MERGED_PW_FILE
    traffic_file = TRAFFIC_DATA_FILE
    final_output_file = FINAL_DATASET_FILE

    # Ensure processed data directory exists
    if not os.path.exists(processed_data_dir):
        logging.error(f"Processed data directory does not exist: {processed_data_dir}")
        raise FileNotFoundError(f"Processed data directory does not exist: {processed_data_dir}")

    num_batches = 20  # Adjust the number of batches as needed

    # Step 1: Split data into batches and prepare scripts
    scripts = split_batch(num_batches, merged_pw_file, traffic_file)

    # Step 2: Run the merge scripts in parallel
    run_scripts_in_folders_concurrently(scripts, max_workers=num_batches)  # Adjust max_workers as needed

    # Step 3: Merge the batch results into a final dataset
    merge_batches(num_batches, final_output_file)


if __name__ == "__main__":
    main_merge_with_traffic()  
