"""
  FileName     [ imdb.py ]
  PackageName  [ final ]
  Synopsis     [ Dataloader of IMDb dataset ]

  Structure:
    IMDb/
    |---val/
    |   |---<movie_name>/
    |   |   |---candidates/
    |   |   |   |---<id>.jpg
    |   |   |---cast/
    |   |   |   |---<id>.jpg
    |---|---<movie_name>/
    ...
"""

import csv
import itertools
import os
import pprint
import random
import time

import numpy as np
# import pandas as pd
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import utils

class IMDbDataset(Dataset):
    def __init__(self, movie_path, feature_path, label_path, transform=None):
        assert ((movie_path is not None) or (feature_path is not None)), "movie_path or feature_path is needed for IMDbDataset"
        
        self.movie_path   = movie_path
        self.feature_path = feature_path
        self.label_path   = label_path

        self.transform = transform
        self.movies_name = os.listdir(movie_path)
        with open(label_path, 'r') as jsonfile:
            self.labels = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        # Get Movie
        movie_name = self.movies_name[index]

        # Cast dict / Candidate dict
        cast_name = os.path.join(self.movie_path, movie_name, 'cast.json')
        candidate_name = os.path.join(self.movie_path, movie_name, 'candidates.json')
        
        label = None
        
        # ---------------------------------------------------
        # To Read the images
        # ---------------------------------------------------
        pass

        # ---------------------------------------------------
        # Features Output dimension: (frames, feature_dim)
        # Images Output dimension:   (frames, channel, height, width)
        # ---------------------------------------------------
        if self.transform:
            image = self.transform(image)

        return image, label

def dataloader_unittest():
    dataset = IMDbDataset("./IMDb/val", None, "./IMDb/val_GT.json", transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    for index, (image, label) in enumerate(dataloader, 1):
        print("Image.shape: {}".format(image.shape))
        print("Label.shape: {}".format(label.shape))

        break

    pass

def main():
    pass

if __name__ == "__main__":
    main()
