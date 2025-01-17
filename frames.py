import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

# Corrected path to the CelebA images
data_dir = r"E:\project\img_align_celeba"

# Image transformation to resize and normalize the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
])

# Custom dataset class
class CelebADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)  # List all image files
        self.labels = [0] * len(self.image_files)  # Labeling all images as "real" (0)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]  # All labels are "real" (0)
        
        return image, label

# Create DataLoader for CelebA dataset
celeba_dataset = CelebADataset(data_dir=data_dir, transform=transform)
train_loader = DataLoader(celeba_dataset, batch_size=32, shuffle=True)







