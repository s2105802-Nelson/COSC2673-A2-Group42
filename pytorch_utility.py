# This file contains reusable classes and functions for Deep Learning with PyTorch on this 
# Cancerous Cells Image Classification problem

import os
from torch.utils.data import Dataset
from torchvision.io import read_image


# Custom DataSet class for loading the image dataset using PyTorch NNs for the isCancerous modelling
class CancerBinaryDataset(Dataset):
    def __init__(self, dfImages, img_dir, transform=None, target_transform=None): 
        # Keep a reference to the data set
        self.df_images = dfImages
        # Set the labels to the the target column in the dataset
        self.img_labels = dfImages["isCancerous"]
        # Pass in the image directory and transform operation
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # Get the length of the dataset from the length of the labels
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # Load the image using the image directory and then from the ImageName col in the dataframe
        img_path = os.path.join(self.img_dir, self.df_images.loc[idx, "ImageName"])
        image = read_image(img_path)

        # Set the label
        label = self.img_labels[idx]

        # Apply the transform
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # Return the image and the label
        return image, label


# Custom DataSet class for loading the image dataset using PyTorch NNs for the CellType modelling
class CancerCellTypeDataset(Dataset):
    def __init__(self, dfImages, img_dir, transform=None, target_transform=None): 
        # Keep a reference to the data set
        self.df_images = dfImages
        # Set the labels to the the target column in the dataset
        self.img_labels = dfImages["cellType"]
        # Pass in the image directory and transform operation
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # Get the length of the dataset from the length of the labels
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # Load the image using the image directory and then from the ImageName col in the dataframe
        img_path = os.path.join(self.img_dir, self.df_images.loc[idx, "ImageName"])
        image = read_image(img_path)

        # Set the label
        label = self.img_labels[idx]

        # Apply the transform
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # Return the image and the label
        return image, label    