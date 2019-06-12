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
    |   |   |---cast.json
    |   |   |---candidate.json
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
import pandas as pd
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import utils

class IMDbTrainset(Dataset):
    def __init__(self, movie_path, feature_path, label_path, transform=None):
        assert ((movie_path is not None) or (feature_path is not None)), "movie_path or feature_path is needed for IMDbDataset"
        
        self.movie_path   = movie_path
        self.feature_path = feature_path
        self.label_path   = label_path
        self.root_path    = os.path.dirname(self.movie_path)

        self.transform = transform

        self.movies         = os.listdir(self.movie_path)
        self.candidate_json = [pd.read_json(os.path.join(self.movie_path, filename, 'candidate.json'), orient='index', typ='series') for filename in os.listdir(self.movie_path)]
        self.cast_json      = [pd.read_json(os.path.join(self.movie_path, filename, 'cast.json'), orient='index', typ='series') for filename in os.listdir(self.movie_path)]

        self.candidates = pd.concat(self.candidate_json, axis=0, keys=self.movies).reset_index()
        self.casts      = pd.concat(self.cast_json, axis=0, keys=self.movies).sort_values().reset_index()
        
        # Total images in dataset
        print("Total candidates in dataset: {}".format(self.candidates.shape))
        # Without "others"
        print("Total casts in dataset:      {}".format(self.casts.shape))
        # print("Total casts in unique: {}".format(self.casts.shape))

        # print(self.candidates)
        # print(self.casts)

    def __len__(self):
        return self.candidates.shape[0]

    def __getitem__(self, index):
        # Get 1 image, label
        candidate, cast = self.candidates.iat[index, 1], self.candidates.iat[index, 2]
        # print(index, candidate, cast)
        
        # ---------------------------------------------------
        # To Read the images
        # ---------------------------------------------------
        image = Image.open(os.path.join(self.root_path, candidate))

        # ---------------------------------------------------
        # Features Output dimension: (feature_dim)
        # Images Output dimension:   (channel, height, width)
        # ---------------------------------------------------
        if self.transform:
            image = self.transform(image)

        return image, cast

def collate_fn(batch):
    """
      To define a function that reads the video by batch.
 
      Params:
      - batch: 
          In pytorch, dataloader generate batch of traindata by this way:
            `self.collate_fn([self.dataset[i] for i in indices])`
          
          In here, batch means `[self.dataset[i] for i in indices]`
          It's a list contains (datas, labels)

      Return:
      - batch: the input tensors in shape (batch, c, h, w)
    """
    # ---------------------------------
    # batch[i][j]
    #   the type of batch[i] is tuple
    # 
    #   i=(0, size) means the batchsize
    #   j=(0, 1) means the data / label
    # ---------------------------------
    
    images = torch.Tensor([x[0] for x in batch])
    
    labels = None
    if batch[0][1] is not None: # If labels exists
        labels = torch.cat([x[1].unsqueeze(0) for x in batch], dim=0)
    
    return images, labels

def dataloader_unittest():
    dataset = IMDbTrainset("./IMDb/val", None, "./IMDb/val_GT.json", transform=transforms.Compose([
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
    dataloader_unittest()

if __name__ == "__main__":
    main()
