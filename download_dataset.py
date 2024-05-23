import gdown
import tarfile
import os

# Google Drive file ID
file_id = '1YMG8z9XTHpAJpXgGE0PymV5vnhOOUFy0'
# Destination file name
destination = 'imagenet-a.tar'

# Download the file from Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)

# Define the destination path for extraction
destination_path = './Datasets/imagenet-a'


# Ensure the destination directory exists
os.makedirs(destination_path, exist_ok=True)

# Extract the tar file
with tarfile.open(destination, 'r') as tar:
    tar.extractall(destination_path)

print("File downloaded and extracted successfully.")