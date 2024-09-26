import os
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def run_scripts_in_folders_concurrently(base_folder, max_workers=20):
    scripts_to_run = []

    # Collect all Python scripts in all folders
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)

        if os.path.isdir(folder_path):
            # Find all Python scripts in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".py"):
                    script_path = os.path.join(folder_path, file_name)
                    scripts_to_run.append(script_path)

    # Run scripts concurrently using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all the scripts to the executor for parallel execution
        futures = {executor.submit(run_script, script): script for script in scripts_to_run}

        # Gather the results as they are completed
        for future in as_completed(futures):
            script_path, time_taken = future.result()
            print(f"Completed: {script_path} in {time_taken}")


if __name__ == '__main__':
    # Provide the base folder where the subfolders are located
    base_folder = "batch"
    run_scripts_in_folders_concurrently(base_folder, max_workers=20)  # Adjust max_workers as needed
