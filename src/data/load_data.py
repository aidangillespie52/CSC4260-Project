# load_data.py

# Imports
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TEST_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/test.csv"))
TRAIN_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/train.csv"))
BOX_TRAIN_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/bbox_train.csv"))
IMAGE_DATA_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/image_data/"))

# Image Dataset for training
class Train_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_DATA_PATH, self.data.iloc[idx, 0])
        label = int(self.data.iloc[idx, 1])

        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def display_image(self, idx):
        # Fetch the image and label
        image, label = self[idx]
        
        # Convert the image to a numpy array and reorder channels
        image = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        
        # Display the image
        plt.imshow(image)
        plt.title(f"Number of Faces: {label}")
        plt.axis('off')  # Hide axes for better visualization
        plt.show()

class Test_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_DATA_PATH, self.data.iloc[idx, 0])
        
        # Open the image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image

    def display_image(self, idx):
        # Fetch the image
        image = self[idx]
        
        # Convert the image to a numpy array and reorder channels
        image = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        
        # Display the image
        plt.imshow(image)
        plt.axis('off')  # Hide axes for better visualization
        plt.show()


def get_train_dataset():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = Train_Dataset(TRAIN_CSV_PATH, transform=transform)

    return dataset

def get_test_dataset():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = Test_Dataset(TEST_CSV_PATH, transform=transform)
    dataset.display_image(20)
    return dataset