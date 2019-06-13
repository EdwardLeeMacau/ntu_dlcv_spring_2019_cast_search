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
    def __init__(self, movie_path, feature_path, label_path, mode, transform=None, debug=False):
        assert ((movie_path is not None) or (feature_path is not None)), "movie_path or feature_path is needed for IMDbDataset"
        assert ((mode == 'classify') or (mode == 'features')), "The parameter 'mode' must be 'classify' or 'feature'"
        
        self.movie_path   = movie_path
        self.feature_path = feature_path
        # self.label_path   = label_path
        self.root_path    = os.path.dirname(self.movie_path)
        
        self.mode = mode
        self.debug = debug

        self.transform = transform

        self.movies         = os.listdir(self.movie_path)
        self.candidate_json = [pd.read_json(os.path.join(self.movie_path, filename, 'candidate.json'), orient='index', typ='series') for filename in os.listdir(self.movie_path)]
        self.cast_json      = [pd.read_json(os.path.join(self.movie_path, filename, 'cast.json'), orient='index', typ='series') for filename in os.listdir(self.movie_path)]

        # Read as pandas.DataFrame and make it
        self.candidates = pd.concat(self.candidate_json, axis=0, keys=self.movies).reset_index()
        self.casts      = pd.concat(self.cast_json, axis=0, keys=self.movies).sort_values().reset_index()

        # add "others" label to self.
        if self.mode == 'classify':
            num_casts = self.casts.shape[0]
            self.casts.loc[num_casts] = ['others', 'no_exist_others.jpg', 'others']

        self.classes = list(self.casts['level_0'])

        if self.mode == 'features':
            self.images = pd.concat((self.candidates, self.casts), axis=0, ignore_index=True)
        
        # print(self.candidates.columns)  # ['level_0', 'level_1', 0]
        # print('level_0 :\n', self.candidates[self.candidates[0] == 'others'])    # 15451 labels of imgs are "others"


        # Total images in dataset
        # print("Total candidates in dataset: {}".format(self.candidates.shape))
        # Without "others"
        # print("Total casts in dataset:      {}\n".format(self.casts.shape))
        # print("Total casts in unique: {}".format(self.casts.shape))

        # print("self.candidates :", self.candidates)
        # print("self.casts :", self.casts)

    @property
    def num_casts(self):
        return self.casts.shape[0]

    def __len__(self):
        if self.mode == 'classify':
            return self.candidates.shape[0]

        if self.mode == 'features':
            return self.images.shape[0]

    def __getitem__(self, index):
        # Get 1 image and label in mode 'classify'
        if self.mode == 'classify':
            image_path, cast = self.candidates.iat[index, 1], self.candidates.iat[index, 2]
        
        # Get 1 image, directory, cast in mode 'features'
        if self.mode == 'features':
            image_path = self.images.iat[index, 1]

        # ---------------------------------------------------
        # To Read the images
        # ---------------------------------------------------
        image = Image.open(os.path.join(self.root_path, image_path))

        # ---------------------------------------------------
        # Features Output dimension: (feature_dim)
        # Images Output dimension:   (channel, height, width)
        # ---------------------------------------------------
        if self.transform:
            image = self.transform(image)

        # string label >> int label
        if self.mode == 'classify':
            label_mapped = self.casts.index[self.casts[0] == cast].to_list()[0] # total : 1 element 
        
            if self.debug:
                print("label_mapped : {} <--> {}".format(label_mapped, cast))
        
        if self.mode == 'features':
            return image

        return image, label_mapped

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

def dataloader_unittest(debug=False):
    print("Classify setting: ")
    
    dataset = IMDbTrainset(
        movie_path = "./IMDb/val",
        feature_path = None,
        label_path = "./IMDb/val_GT.json",
        mode = 'classify',
        debug = debug,
        transform = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print("Length of dataset: {}".format(len(dataset)))

    for index, (image, label) in enumerate(dataloader, 1):
        print("Image.shape: {}".format(image.shape))
        print("Label.shape: {}".format(label.shape))
        # print("Label: {}".format(label))
        print()

        # if "others" in label:
        # if 198 in label:   # "others" mapped to 198
        #     print('dataloader unitest finished, has 198("others") in labels.')
        break

    #################################################################################

    print("Features setting: ")

    dataset = IMDbTrainset(
        movie_path = "./IMDb/val",
        feature_path = None,
        label_path = "./IMDb/val_GT.json",
        mode = 'features',
        debug = debug,
        transform = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print("Length of dataset: {}".format(len(dataset)))

    for index, (image) in enumerate(dataloader, 1):
        print("Image.shape: {}".format(image.shape))
        # print("Label.shape: {}".format(label.shape))
        # print("Label: {}".format(label))
        print()

        # if "others" in label:
        # if 198 in label:   # "others" mapped to 198
        #     print('dataloader unitest finished, has 198("others") in labels.')
        break


if __name__ == "__main__":
    dataloader_unittest()
