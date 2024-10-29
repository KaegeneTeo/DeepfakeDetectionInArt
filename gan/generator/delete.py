import os

# Define the base directory where all the folders are located
base_directory = "C:/Users/kaege/OneDrive/Desktop/SMU/Year 3/Project/similar/inpainting"

# Loop through each subdirectory in the base directory
for folder_name in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Loop through each file in the current folder
        for file_name in os.listdir(folder_path):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)

            # Delete the file if it's not "original.png"
            if file_name != "original.png":
                print(f"Deleting {file_path}")
                os.remove(file_path)

print("Cleanup completed. Only 'original.png' files are kept in each folder.")
