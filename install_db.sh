# curl -O https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar

# echo "Dataset successfully downloaded"

# mkdir -p datasets

# echo "Created directory /datasets"

# tar -xf imagenet-a.tar -C datasets

# echo "Dataset successfully extracted"

#!/bin/bash

# Check current user
current_user=$(whoami)
echo "Current user: $current_user"

# Ensure the directory is not in use
echo "Checking if directory is in use..."
lsof +D DeepLearning_project

# Change ownership and permissions
echo "Changing ownership and permissions..."
sudo chown -R $current_user:$current_user DeepLearning_project
sudo chmod -R u+w DeepLearning_project

# Remove the directory
echo "Removing directory..."
sudo rm -rf DeepLearning_project

echo "Directory removed successfully."
