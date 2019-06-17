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

        self.movies         = sorted(os.listdir(self.movie_path))
        self.candidate_json = [pd.read_json(os.path.join(self.movie_path, filename, 'candidate.json'), orient='index', typ='series') 
                                  for filename in self.movies]
        self.cast_json      = [pd.read_json(os.path.join(self.movie_path, filename, 'cast.json'), orient='index', typ='series') 
                                  for filename in self.movies]

        # Read as pandas.DataFrame and make it
#        self.candidates = pd.concat(self.candidate_json, axis=0, keys=self.movies).reset_index()
#        self.casts      = pd.concat(self.cast_json, axis=0, keys=self.movies).reset_index()

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

class TripletDataset(Dataset):
    def __init__(self, root_path, data_path, moviename, mode='classify', keep_others=True, transform=None, debug=False):
        
        self.root_path = root_path  # IMDb
        self.data_path = data_path  # IMDb/train
        self.moviename = moviename
        self.mode = mode
        self.keep_others = keep_others
        self.debug = debug

        self.transform = transform
        
        self.movies = os.listdir(self.data_path)
        
#        data_path = 'IMDb/train'
#        moviename = 'tt6518634'
#        cast/"
        # Read json as pandas.DataFrame and divide candidates and others
        candidate_json = pd.read_json(os.path.join(data_path, moviename, 'candidate.json'),
                                           orient='index', typ='series').reset_index()
        if keep_others:
            self.candidates = candidate_json[candidate_json[0] != "others"]
            self.others = candidate_json[candidate_json[0] == "others"]
        else:
            self.candidates = candidate_json
            
        self.casts = pd.read_json(os.path.join(data_path, moviename, 'cast.json'),
                                           orient='index', typ='series') .reset_index()
        num_casts = self.casts.shape[0]
        if keep_others:
            self.casts.loc[num_casts] = ['no_exist_others.jpg', 'others']

        if self.mode == 'features':
            self.images = pd.concat((self.candidates, self.casts), axis=0, ignore_index=True)

#        self.classes = list(self.casts[0])

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
        if self.mode == 'classify' or self.mode == 'faces':
            return self.candidates.shape[0]

        if self.mode == 'features':
            return self.images.shape[0]

    def __getitem__(self, index):
        # -------------------------------------------------
        # Mode:
        #   Classify: 
        #     get 1 image and 1 label
        #   Faces: get 1
        #     get 1 image, label is 1 if it contains a face
        #   Features:
        #     get 1 image only.
        # -------------------------------------------------
        if self.mode == 'classify':
            image_path, cast = self.candidates.iat[index, 0], self.candidates.iat[index, 1]

        if self.mode == 'faces':
            image_path, cast = self.candidates.iat[index, 0], self.candidates.iat[index, 1]

        if self.mode == 'features':
            image_path = self.images.iat[index, 0]

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
        
        if self.mode == 'faces':
            label_mapped = (cast != 'others')

            if self.debug:
                print("label_mapped : {} <--> {}".format(label_mapped, cast))
            
        if self.mode == 'features':
            return image

        return image, label_mapped
    
class CastDataset(Dataset):
    def __init__(self, root_path, data_path, mode='classify', keep_others=True, transform=None, debug=False):
        
        self.root_path = root_path  # IMDb
        self.data_path = data_path  # IMDb/train 
        self.mode = mode
        self.keep_others = keep_others
        self.debug = debug

        self.transform = transform
        
        self.movies = os.listdir(self.data_path)
        
#        data_path = 'IMDb/train'
#        moviename = 'tt6518634'
#        cast/"
        
    @property
    def num_casts(self):
        
        return self.casts.shape[0]

    def __len__(self):
        
        return len(self.movies)


    def __getitem__(self, index):
        
        moviename = self.movies[index]
        # Read json as pandas.DataFrame and divide candidates and others
        candidate_json = pd.read_json(os.path.join(self.data_path, moviename, 'candidate.json'),
                                           orient='index', typ='series').reset_index()
        if self.keep_others:
            others = candidate_json[candidate_json[0] == "others"]
            rn = torch.randint(0, len(others), (1,)).tolist()[0]
        
        casts = pd.read_json(os.path.join(self.data_path, moviename, 'cast.json'),
                                           orient='index', typ='series').reset_index()
        num_casts = casts.shape[0]
        if self.keep_others:
            casts.loc[num_casts] = [others.iat[rn,0], 'others']
        
        # -------------------------------------------------
        # Mode:
        #   Classify: 
        #     get 1 image and 1 label
        #   Features:
        #     get 1 image only.
        # -------------------------------------------------
        
        images = torch.tensor([])
        labels = []
        for idx in range(num_casts+1):
            if self.mode == 'classify':
                image_path, cast = casts.iat[idx, 0], casts.iat[idx, 1]

            if self.mode == 'features':
                image_path = self.images.iat[idx, 0]
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
            images = torch.cat((images,image.unsqueeze(0)), dim=0)

            # string label >> int label
            if self.mode == 'classify':
                label_mapped = casts.index[casts[0] == cast] # total : 1 element 
            
                if self.debug:
                    print("label_mapped : {} <--> {}".format(label_mapped, cast))
            
                labels.append(label_mapped)
            
        if self.mode == 'features':
            return images
        
        print(num_casts)
        print(images.size())
        print(labels)
        
        return images, labels, moviename

def load_candidate(root_path, datapath, bsize):
    
    movie_list = os.listdir(datapath)
    all_dataset = {}
    all_loader = {}
    info = {}
    info['len'] = []
    transform1 = transforms.Compose([
                        transforms.Resize((448,448), interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                             ])
    if datapath == 'IMDb/train':
        keep_other = True
        shuf = True
    else:
        keep_other = False
        shuf = False
    
    for mov in movie_list:
        num_cast = len(os.listdir(datapath + '/' + mov))
        all_dataset[mov] = TripletDataset(root_path,
                   datapath,
                   mov,
                   mode='classify',
                   keep_others=keep_other,
                   transform=transform1,
                   debug=False)
        all_loader[mov] = DataLoader(all_dataset[mov],
                            batch_size=16-num_cast,
                            shuffle=shuf,
                            num_workers=0)
        info['len'].append(len(all_dataset[mov]))
    
    return all_dataset, all_loader


def dataloader_unittest(debug=False):
    print("Classify setting: ")
#    (self, data_path, moviename, mode='classify', keep_others=True, transform=None, debug=False):
        
    dataset = TripletDataset(
        data_path = "./IMDb/train",
        moviename = "tt6518634",
        mode = 'classify',
        debug = True,
        transform = transforms.Compose([
                transforms.Resize((448,448), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ]))

    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=0)

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
'''
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
'''

if __name__ == "__main__":
    dataloader_unittest()
