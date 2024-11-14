import papermill as pm
import os

# This ensures that executive.py runs from the /src directory
os.chdir('Group B/src/')


# Folder path where the CSV files are located
folder_path_auto_run = '../data/spawning csv files'  

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path_auto_run) if f.endswith('.csv')]

# Loop through each file and run the notebook with that CSV file
for file_name in csv_files:
    file_path_auto_run = os.path.join(folder_path_auto_run, file_name)
    
    # Define the output notebook file name 
    output_notebook = f"output_{file_name.replace('.csv', '')}.ipynb"
    
    # Run the notebook using papermill, passing the CSV file path as a parameter
    pm.execute_notebook(
        'park_simulation.ipynb',              # Input notebook
        output_notebook,               # Output notebook
        parameters=dict(csv_file=file_path_auto_run)  # Pass the CSV file path as a parameter
    )
    
    print(f"Executed notebook for {file_name}, output saved to {output_notebook}")
