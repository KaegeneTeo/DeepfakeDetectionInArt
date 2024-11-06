import os

# Define the base directory where all the folders are located
base_directory = "/common/home/users/h/haotian.hu.2021/DeepfakeDetectionInArt/stable_diffusionV2/generator/output"

# Loop through each subdirectory in the base directory
for folder_name in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Loop through each file in the current folder
        for file_name in os.listdir(folder_path):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)

            # Delete the file if it contains "group" in the filename
            if "group" in file_name:
                print(f"Deleting {file_path}")
                os.remove(file_path)


print("Cleanup completed.")
