path = "/home/andreafabbricatore/rainbot/datasets/final_datasets"

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

def upload_to_wasabi():
    # Walk through the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # Get the full local path
            local_path = os.path.join(root, file)
            
            # Create the S3 key (path in bucket) by removing the base path
            s3_key = os.path.join("truly_final_datasets", os.path.relpath(local_path, path))
            
            # Upload the file
            print(f"Uploading: {local_path} -> {s3_key}")
            s3.upload_file(local_path, BUCKET_NAME, s3_key)

# Execute the upload
upload_to_wasabi()

