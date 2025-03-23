import os
import pandas as pd
import shutil

# Define paths
metadata_path = "D:/IPPR_PBL/archive/HAM10000_metadata.csv"
image_folders = ["D:/IPPR_PBL/archive/HAM10000_images_part_1", "D:/IPPR_PBL/archive/HAM10000_images_part_2"]
dataset_folder = "D:/IPPR_PBL/dataset/"

# Read metadata CSV
df = pd.read_csv(metadata_path)

# Create destination folders if they don’t exist
benign_dir = os.path.join(dataset_folder, "benign")
malignant_dir = os.path.join(dataset_folder, "malignant")
os.makedirs(benign_dir, exist_ok=True)
os.makedirs(malignant_dir, exist_ok=True)

# Define benign and malignant labels
benign_labels = ["nv", "bkl"]  # Benign types
malignant_labels = ["mel"]  # Malignant types

# Move images
missing_files = 0
for _, row in df.iterrows():
    image_id = row["image_id"]
    label = row["dx"]  # Diagnosis
    
    # Search for image in both folders
    src_path = None
    for folder in image_folders:
        temp_path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(temp_path):
            src_path = temp_path
            break  # Stop searching once found
    
    if src_path is None:  # Debugging: Missing files
        print(f"❌ Missing: {image_id}.jpg")
        missing_files += 1
        continue
    
    # Assign to category folder
    if label in benign_labels:
        dest_path = os.path.join(benign_dir, f"{image_id}.jpg")
    elif label in malignant_labels:
        dest_path = os.path.join(malignant_dir, f"{image_id}.jpg")
    else:
        continue  # Skip unknown labels

    shutil.move(src_path, dest_path)  # Move the file
    print(f"✅ Moved: {src_path} -> {dest_path}")

print(f"🎯 Image organization complete! Missing files: {missing_files}")
