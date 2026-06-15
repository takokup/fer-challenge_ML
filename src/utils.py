import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class FERDataset(Dataset):
    def __init__(self, csv_path, split='Training', transform=None, subset_fraction=1.0):
        df = pd.read_csv(csv_path)
        df = df[df['Usage'] == split].reset_index(drop=True)
        if subset_fraction < 1.0:
            df = df.sample(frac=subset_fraction, random_state=42).reset_index(drop=True)
        self.data = df
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        pixels = list(map(int, self.data.iloc[idx]['pixels'].split()))
        img = np.array(pixels, dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(self.data.iloc[idx]['emotion'])
        return img, label

def get_dataloaders(csv_path, batch_size=64, subset_fraction=1.0, use_augmentation=True):
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_set = FERDataset(csv_path, 'Training', train_transform, subset_fraction)
    val_set = FERDataset(csv_path, 'PublicTest', val_transform, subset_fraction)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    return train_loader, val_loader
