import os
import time
import requests

######## MAKE SURE THE ARGUMENTS ARE CORRECT ########
VERSION = "4.0" # Make sure the version is correct
ROOT_FOLDER ='./Data_Download_Test'  # Set your desired data folder here
######## MAKE SURE THE ARGUMENTS ARE CORRECT ########

SERVER_URL = "https://dataverse.ucla.edu"
DATASET_ID = "42639" # Make sure correct id
DELAY_BETWEEN_UPLOADS = 10
MAX_RETRIES = 5

def list_and_download_files(dataset_id, version):
    """Lists all files in a given dataset version and downloads each one into its respective directory structure."""
    url = f"{SERVER_URL}/api/datasets/{dataset_id}/versions/{version}/files"
    response = requests.get(url)
    
    if response.status_code == 200:
        files = response.json().get('data', [])
        
        for file in files:
            persistent_id = file['dataFile']['persistentId']
            filename = file['dataFile']['filename']
            directory_label = file['directoryLabel']  
            
            # Create the directory structure if it doesn't exist
            if directory_label:
                full_directory_path = os.path.join(ROOT_FOLDER, directory_label)
                if not os.path.exists(full_directory_path):
                    os.makedirs(full_directory_path)

                full_file_path = os.path.join(full_directory_path, filename)
                
                print(f"File ID: {file['dataFile']['id']}, Filename: {filename}, Persistent ID: {persistent_id}, Download Path: {full_file_path}")
                
                # Download each file with retry logic
                download_file(persistent_id, full_file_path)
                
                # Sleep for 10 seconds between downloads
                time.sleep(DELAY_BETWEEN_UPLOADS)
            else:
                print(f"File does not have a directory label defined: {filename}")
    else:
        print(f"Failed to list files: {response.status_code}, {response.text}")

def download_file(persistent_id, full_file_path, max_retries=MAX_RETRIES):
    url = f"{SERVER_URL}/api/access/datafile/:persistentId?persistentId={persistent_id}"
    for attempt in range(max_retries):
        response = requests.get(url)
        
        if response.status_code == 200:
            # Save the file to the specified full file path
            with open(full_file_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded: {full_file_path}")
            return 
        else:
            print(f"Attempt {attempt + 1}: Failed to download file {persistent_id} - {response.status_code}, {response.text}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print(f"Max retries reached for file: {persistent_id}")

if __name__ == "__main__":
    print("IMPORTANT: Make sure ALL arguments are correct at the beginning of this file!!!")
    print("Listing and downloading files in the dataset...")
    list_and_download_files(DATASET_ID, VERSION)
