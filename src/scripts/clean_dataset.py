# clean_train_dataset.py

# Imports
import os
import pandas as pd

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAIN_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/train.csv"))
IMAGE_DATA_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/image_data/"))

# Load training CSV
df = pd.read_csv(TRAIN_CSV_PATH)

# Get list of valid image filenames from CSV (assuming the first column contains image names)
valid_images = set(df.iloc[:, 0].tolist())

# Iterate over image files in the directory
for image_file in os.listdir(IMAGE_DATA_PATH):
    image_path = os.path.join(IMAGE_DATA_PATH, image_file)

    # Check if file is an actual image and not in the valid list
    if os.path.isfile(image_path) and image_file not in valid_images:
        os.remove(image_path)  # Delete the unlisted image
        print(f"Removed: {image_file}")

print("Cleanup complete! Images not in train.csv removed.")
