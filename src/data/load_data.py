# load_data.py

# Imports

# Needed for grabbing local python files
import os

# Needed for ML
import torch
import pandas as pd

# Used for creating dataset and changing it as necessary
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Needed for plotting images
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TEST_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/test.csv"))
TRAIN_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/train.csv"))
BOX_TRAIN_CSV_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/bbox_train.csv"))
IMAGE_DATA_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data/image_data/"))

data = pd.read_csv(BOX_TRAIN_CSV_PATH)

# Ensure that all files within img folder actually exist
data = data[data['Name'].apply(
    lambda img: os.path.exists(os.path.join(IMAGE_DATA_PATH, img))
)].reset_index(drop=True)

unique_imgs = data['Name'].unique()

class ObjectDetectionDataset(Dataset):
    def __init__(self, df, unique_imgs, indices):
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_name = self.unique_imgs[self.indices[idx]]
        boxes = self.df[self.df.Name == img_name][["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")
        #print(f"{img_name}: {boxes.shape}, {boxes}")
        img = Image.open(os.path.join(IMAGE_DATA_PATH, img_name)).convert('RGB')
        labels = torch.ones((boxes.shape[0]) , dtype= torch.int64)
        target = {}
        target["boxes"] = torch.tensor(boxes)
        target["label"] = labels
        target["img_index"] = torch.tensor(self.indices[idx])
        return transforms.ToTensor()(img), target

train_inds, val_inds = train_test_split(range(unique_imgs.shape[0]), test_size=0.1)

def custom_collate(data):
    return data

train_dl = DataLoader(ObjectDetectionDataset(data, unique_imgs, train_inds),
                    batch_size=2,
                    shuffle=True,
                    collate_fn=custom_collate,
                    pin_memory=True if torch.cuda.is_available() else False
                    )

val_dl = DataLoader(ObjectDetectionDataset(data, unique_imgs, train_inds),
                    batch_size=1,
                    shuffle=True,
                    collate_fn=custom_collate,
                    pin_memory=True if torch.cuda.is_available() else False
                    )

class FaceCountDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame with columns ['image_name', 'num_faces']
            image_dir (str): Path to directory containing images
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx]['Name'])
        image = Image.open(img_name).convert('RGB')
        num_faces = torch.tensor(self.df.iloc[idx]['HeadCount'], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, num_faces, img_name

# Example usage
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def create_dataloader(df, image_dir, batch_size=1, shuffle=True):
    dataset = FaceCountDataset(df, image_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Splitting dataset into training and validation
train_df, val_df = train_test_split(pd.read_csv(TRAIN_CSV_PATH), test_size=0.1, random_state=42)

# Creating DataLoader instances
simple_train_dl = create_dataloader(train_df, IMAGE_DATA_PATH)
simple_val_dl = create_dataloader(val_df, IMAGE_DATA_PATH, shuffle=False)