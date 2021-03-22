"""Class for custom Youtube thumbnails image dataset
"""
import os
import pandas
import glob
from PIL import Image
from torch.utils.data import Dataset

class YoutubeDataset(Dataset):
    """Youtube image dataset. This contains no labels.
    """
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_list = [f for f in glob.glob(self.img_dir + '/*.jpg') if os.path.isfile(f)]

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)

        return image, idx, img_path
