# load_data.py

# Imports

# Needed for grabbing local python files
import os

# Needed for ML
import torch
import pandas as pd

# Used for creating dataset and changing it as necessary
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Needed for plotting images
import matplotlib.pyplot as plt
from PIL import Image

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TEST_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/test.csv"))
TRAIN_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/train.csv"))
BOX_TRAIN_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/bbox_train.csv"))
IMAGE_DATA_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/image_data/"))

class Train_Dataset(Dataset):
    def __init__(self, transform=None):
        self.data = pd.read_csv(TRAIN_CSV_PATH)
        self.transform = transform

        # Ensure that all files within img folder actually exist
        self.data = self.data[self.data['Name'].apply(
            lambda img: os.path.exists(os.path.join(IMAGE_DATA_PATH, img))
        )].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_DATA_PATH, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        
        label = float(self.data.iloc[idx, 1])  
        label = torch.tensor(label, dtype=torch.float32)

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
    def __init__(self, transform=None):
        self.data = pd.read_csv(TEST_CSV_PATH)
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

class ObjectDetectionDataset(Dataset):
    def __init__(self, transforms=None):
        self.annotations = pd.read_csv(BOX_TRAIN_CSV_PATH)
        self.image_dir = IMAGE_DATA_PATH
        self.transforms = transforms

        self.annotations = self.annotations[self.annotations['Name'].apply(
            lambda img: os.path.exists(os.path.join(self.image_dir, img))
        )]

        self.image_filenames = self.annotations['Name'].unique()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(img_path).convert("RGB")

        # Filter annotations for this image
        img_annotations = self.annotations[self.annotations['Name'] == image_filename]
        boxes = img_annotations[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Assign dummy labels (assuming 1 class, modify if needed)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        # Required fields for torchvision models
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms:
            image = self.transforms(image)

        return image, target

def get_train_dataset_with_cords():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    return ObjectDetectionDataset(transforms=transform)

def get_train_dataset():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = Train_Dataset(transform=transform)

    return dataset

def get_test_dataset():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = Test_Dataset(TEST_CSV_PATH, transform=transform)
    dataset.display_image(20)
    return dataset