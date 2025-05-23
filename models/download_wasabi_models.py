import os
import boto3
from botocore.config import Config

# -------- CONFIGURATION --------
WASABI_ACCESS_KEY = "VIH97PD2FNFFMYQAESDR"
WASABI_SECRET_KEY = "X2UgOtGwJP5DFYnYjSloe9ilVVb89wseGiRKZa9T"
WASABI_REGION = "eu-south-1"  # or your region
BUCKET_NAME = "geoguesser-dataset"
DEST_DIR = "./datasets"  # Local destination folder
# --------------------------------

# Create session with Wasabi (S3-compatible)
session = boto3.session.Session()

s3 = session.client(
    service_name='s3',
    region_name=WASABI_REGION,
    aws_access_key_id=WASABI_ACCESS_KEY,
    aws_secret_access_key=WASABI_SECRET_KEY,
    endpoint_url=f'https://s3.{WASABI_REGION}.wasabisys.com',  # Wasabi endpoint
    config=Config(signature_version='s3v4')
)

def download_models():
    # Define the folders we want to download
    folders_to_download = ["models"]
    
    # Create destination directory if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # List all objects in the bucket
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME)
    
    # Download each file
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Check if the file is in one of our target folders
            if any(key.startswith(folder) for folder in folders_to_download):
                # Create the full local path
                local_path = os.path.join(DEST_DIR, key)
                # Create any necessary subdirectories
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                # Download the file
                print(f"Downloading: {key} -> {local_path}")
                s3.download_file(BUCKET_NAME, key, local_path)

download_models()