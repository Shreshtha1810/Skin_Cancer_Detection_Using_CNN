import os
import shutil
import random

# Define paths
source_dir = "D:/IPPR_PBL/dataset"  # This should contain "benign" and "malignant" folders
train_dir = "D:/IPPR_PBL/dataset/train"
val_dir = "D:/IPPR_PBL/dataset/val"

# Create train/val directories
for category in ["benign", "malignant"]:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

# Split function
def split_data(source, train_dest, val_dest, split_ratio=0.8):
    images = os.listdir(source)
    random.shuffle(images)  # Shuffle images
    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    # Move images to respective folders
    for img in train_images:
        shutil.move(os.path.join(source, img), os.path.join(train_dest, img))

    for img in val_images:
        shutil.move(os.path.join(source, img), os.path.join(val_dest, img))

# Apply function to both classes
split_data(os.path.join(source_dir, "benign"), os.path.join(train_dir, "benign"), os.path.join(val_dir, "benign"))
split_data(os.path.join(source_dir, "malignant"), os.path.join(train_dir, "malignant"), os.path.join(val_dir, "malignant"))

print("✅ Dataset successfully split into train and validation sets!")
