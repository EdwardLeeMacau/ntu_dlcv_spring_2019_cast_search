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

class TripletDataset(Dataset):
    def __init__(self, root_path, data_path, mode='classify', drop_others=True, transform=None, debug=False, action='train'):
        
        self.root_path = root_path  # IMDb
        self.data_path = data_path  # IMDb/train
        self.mode = mode
        self.drop_others = drop_others

        self.transform = transform
        self.movies = os.listdir(self.data_path)
        self.mv = ''
        self.action = action
        # data_path = 'IMDb/train'
        # moviename = 'tt6518634'
        # cast/"
        self.all_data = {}
        for mov in self.movies:
            # Read json as pandas.DataFrame and divide candidates and others
            candidate_json = pd.read_json(os.path.join(data_path, mov, 'candidate.json'),
                                               orient='index', typ='series').reset_index()
            self.casts = pd.read_json(os.path.join(data_path, mov, 'cast.json'),
                                               orient='index', typ='series') .reset_index()
            num_casts = self.casts.shape[0]
          
            if not drop_others:
                self.casts.loc[num_casts] = ['no_exist_others.jpg', 'others']
                self.candidates = candidate_json
            else:
                self.candidates = candidate_json[candidate_json[0] != "others"]
           
            self.all_data[mov] = [ self.candidates, self.casts ]     
            

    @property
    def num_casts(self):
        return self.casts.shape[0]

    def __len__(self):
        if self.mode == 'classify' :
            return self.candidates.shape[0]

    def __getitem__(self, idx):
        
        if self.action == 'test':    
            """
            TODO: 
            """
            pass
        else:
            pass
        casts = self.all_data[self.mv][1]
        candidates = self.all_data[self.mv][0]
        index = int(torch.randint(0, len(candidates[0]), (1,)).tolist()[0])

        # print(casts)
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
            image_path, cast = candidates.iat[index, 0], candidates.iat[index, 1]

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
            label_mapped = casts.index[casts[0] == cast].tolist()[0] 
            # total : 1 element 
        

        return image, label_mapped, index
    
class CastDataset(Dataset):
    def __init__(self, root_path, data_path, mode='classify', drop_others=True, transform=None, debug=False, action='train'):
        
        self.root_path = root_path  # IMDb
        self.data_path = data_path  # IMDb/train 
        self.mode = mode
        self.drop_others = drop_others
        self.debug = debug
        self.action = action

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
        # print(moviename)

        if self.action == 'test':
            cast_folder = os.path.join(self.data_path, moviename, 'cast')
            candidates_folder = os.path.join(self.data_path, moviename, 'candidates')
            
            casts = os.listdir(cast_folder)
            candidates = os.listdir(candidates_folder)
            num_casts = len(casts)

            """
            TODO:
                return (imgs, moviename)
            """


        
        else:
            # Read json as pandas.DataFrame and divide candidates and others
            casts = pd.read_json(os.path.join(self.data_path, moviename, 'cast.json'),
                                                orient='index', typ='series') .reset_index()
            candidate_json = pd.read_json(os.path.join(self.data_path, moviename, 'candidate.json'),
                                            orient='index', typ='series').reset_index()
            num_casts = casts.shape[0]

            if not self.drop_others:
                others = candidate_json[candidate_json[0] == "others"]
                rn = int(torch.randint(0, len(others), (1,)).tolist()[0])
                casts.loc[num_casts] = [others.iat[rn,0], 'others']

            self.casts = casts

            # if not self.drop_others:
                # casts.loc[num_casts] = [others.iat[rn,0], 'others']
            # else:
                # casts.loc[num_casts] = ['no_this_img.jpg', 'others']
            
            # -------------------------------------------------
            # Mode:
            #   Classify: 
            #     get 1 image and 1 label
            #   Features:
            #     get 1 image only.
            # -------------------------------------------------
            
            images = torch.tensor([])
            labels = torch.tensor([],dtype=torch.long)
            if not self.drop_others:
                num_casts += 1
            if self.debug:
                print("num_casts in CastDataset :", num_casts)
                
            for idx in range(num_casts):
                if self.mode == 'classify':
                    image_path, cast = casts.iat[idx, 0], casts.iat[idx, 1]

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
                    label_mapped = casts.index[casts[0] == cast].tolist() # total : 1 element 
                    label_mapped = torch.tensor(label_mapped)
                    # print(label_mapped)
                    labels = torch.cat((labels,label_mapped),dim=0)
                
            # print(num_casts)
            # print(images.size())
            # print(labels)
            # print(moviename)
                    
            return images, labels, moviename

def dataloader_unittest(debug=False):
    print("Classify setting: ")
#    (self, data_path, moviename, mode='classify', drop_others=True, transform=None, debug=False):
        
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
