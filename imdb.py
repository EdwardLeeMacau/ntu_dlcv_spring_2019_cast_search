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
        | ...
        |
        |---test/
        |   |---<movie_name>/
        |   |   |---candidates/
        |   |   |   |---<id>.jpg
        |   |   |---cast/
        |   |   |   |---<id>.jpg
        |---|---<movie_name>/
        ...

        IMDb/
        |---test/
        |   |---<movie_name>/
        |   |   |---candidates/
        |   |   |   |---features.npy
        |   |   |   |---names.npy
        |   |   |   |---labels.npy
        |   |   |---cast/
        |   |   |   |---features.npy
        |   |   |   |---names.npy
        |   |   |   |---labels.npy
        ...

    Warn : 
        Dataset : when action = 'save' , will parse all json files recursicely,
                    so only used when datapath given '<root>/train/' or '<root>/val/'
                    '<root>/test' has to save_features through inference_csv.py

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

# To pop the candidates
class CandDataset(Dataset):
    def __init__(self, data_path, drop_others=True, transform=None, debug=False, action='train', load_feature=False):
        '''
        - drop_others : only used when aciton = 'train', optional ('val', 'test', 'save' will not drop anyway)
        '''
        assert action in ('train', 'val', 'test', 'save'), "Wrong params 'action'"

        self.root_path = os.path.dirname(data_path) # IMDb
        self.data_path = data_path                  # IMDb/train
        self.drop_others = drop_others
        
        self.transform = transform
        self.movies = os.listdir(self.data_path)
        self.mv = self.movies[0]    # initialize(avoid '' keyerror when dataloader initialize)
        self.action = action
        self.load_feature = load_feature

        if load_feature:
            '''
            TODO : 
            load from .npy file
            '''
            pass

        if action in ('train'):
            self.all_candidates = {}
            self.all_casts = {}
            
            for mov in self.movies:
                # Read json as pandas.DataFrame and divide candidates and others
                candidate_json = pd.read_json(os.path.join(data_path, mov, 'candidate.json'),
                                    orient='index', typ='series').reset_index()
                casts = pd.read_json(os.path.join(data_path, mov, 'cast.json'),
                                    orient='index', typ='series').reset_index()
                num_casts = casts.shape[0]
            
                if not drop_others:
                    # add label "others" to casts
                    # casts.loc[num_casts] = ['no_exist_others.jpg', 'others']

                    # dont actually add "others" to casts, but handle at __getietm__()
                    candidates = candidate_json
                else:
                    # remove label "others" in origin candidate_json
                    candidates = candidate_json[candidate_json[0] != "others"]
            
                # self.all_data[mov] = [ candidates, casts ]
                self.all_candidates[mov] = candidates
                self.all_casts[mov] = casts
        
        elif action in ('save', 'val'):
            self.all_candidates = {}
            self.all_casts = {}
            
            for mov in self.movies:
                # Read json as pandas.DataFrame and divide candidates and others
                candidate_json = pd.read_json(os.path.join(data_path, mov, 'candidate.json'),
                                    orient='index', typ='series').reset_index()
                casts = pd.read_json(os.path.join(data_path, mov, 'cast.json'),
                                    orient='index', typ='series').reset_index()
                
                self.all_candidates[mov] = candidate_json
                self.all_casts[mov] = casts       

        elif action in 'test':
            pass

    def __len__(self):
        if self.action == 'train':
            return len(self.all_candidates[self.mv])
        
        elif self.action in ['test', 'save', 'val']:
            return self.leng

    def set_mov_name_train(self, mov):
        self.mv = mov
        # self.leng = len(self.all_casts[self.mv])

    def set_mov_name_val(self, mov):
        self.mv = mov
        self.candidate_df = self.all_candidates[mov]    # dataframe.columns = ['index', 0] (files, label)
        self.cast_df = self.all_casts[mov]              # dataframe.columns = ['index', 0] (files, label)
        self.leng = len(self.candidate_df)

    def set_mov_name_save(self, mov):
        self.set_mov_name_val(mov)

    def set_mov_name_test(self, mov):
        self.mv = mov
        self.movie_path = os.path.join(self.data_path, mov) # to get image path
        self.candidate_file_list = os.listdir(os.path.join(self.movie_path, 'candidates'))
        self.leng = len(self.candidate_file_list)

    def __getitem__(self, idx):
        if self.action == 'test':
            '''
            Return (old, indexing by movie):
              - images (torch.tensor) : all candidates transformed images 
              - file_name_list (list) : all candidates img file name (list of str (no ".jpg") )
                                    ['tt1840309_0000', 'tt1840309_0001', ...]

            Return (new, indexing by img):
              - image (torch.tensor) : transformed image
              - img_name (str) : img file name (list of str (no ".jpg")
            '''
            candidate_file = self.candidate_file_list[idx]
            image_path = os.path.join(self.movie_path, 'candidates', candidate_file)
            image = Image.open(image_path)

            if self.transform:
                image = self.transform(image)

            img_name = candidate_file[:-4]    # remove ".jpg"
            return image, img_name

        elif self.action in ('save', 'val'):
            '''
              Return candidates 1 by 1.
              (indexing by img)
              
              Return :
                - image (torch.tensor) : transformed image
                - label_mapped (int)
                - img_name (str) : img file name (list of str (no ".jpg")
            '''
            image_path, label_str = self.candidate_df.iat[idx, 0], self.candidate_df.iat[idx, 1]
            img_name = image_path.split('/')[-1].split('.')[0]

            image = Image.open(os.path.join(self.root_path, image_path))
            if self.transform:
                image = self.transform(image)

            num_casts = self.cast_df.shape[0]
            # string label >> int label
            label_mapped = self.cast_df.index[self.cast_df[0] == label_str].tolist()
            # handle : if no "others" in self.cast_df, return num_casts as label (old labes : 0 ~ self.num_casts - 1)
            label_mapped = label_mapped[0] if len(label_mapped) > 0 else num_casts
            
            if self.action == 'save':
                return image, label_mapped, img_name         
            if self.action == 'val':
                return image, label_mapped, idx

        elif self.action == 'train':
            '''
              Return random candidate.
              (idx is unrelated to the output image)

              Return:
              - image
              - label_mapped
              - index
            '''
            casts = self.all_casts[self.mv]
            candidates = self.all_candidates[self.mv]

            # randomly generate an index to get candidate image
            index = int(torch.randint(0, len(candidates[0]), (1,)).tolist()[0])

            image_path, cast = candidates.iat[index, 0], candidates.iat[index, 1]
            image = Image.open(os.path.join(self.root_path, image_path))
            if self.transform:
                image = self.transform(image)

            # string label >> int label
            label_mapped = casts.index[casts[0] == cast].tolist()[0] 
            # print('label check : [{} >> {}]'.format(cast, label_mapped))
            
            return image, label_mapped, index

# To pop the cast images
class CastDataset(Dataset):
    def __init__(self, data_path, drop_others=True, transform=None, debug=False, action='train', load_feature=False):
        '''
        - drop_others : only used when aciton = 'train', optional ('val', 'test', 'save' will not drop anyway)
        '''
        assert action in ('train', 'val', 'test', 'save'), "Wrong params 'action'"

        self.root_path = os.path.dirname(data_path) # IMDb
        self.data_path = data_path                  # IMDb/train 
        self.drop_others = drop_others
        self.debug = debug
        self.action = action
        self.transform = transform
        self.movies = os.listdir(self.data_path)    # moviename = 'tt6518634'
        self.load_feature = load_feature

        if action in ('train'):
            self.all_candidates = {}
            self.all_casts = {}
            for mov in self.movies:
                # Read json as pandas.DataFrame and divide candidates and others
                candidate_json = pd.read_json(os.path.join(data_path, mov, 'candidate.json'),
                                    orient='index', typ='series').reset_index()
                casts = pd.read_json(os.path.join(data_path, mov, 'cast.json'),
                                    orient='index', typ='series').reset_index()
                num_casts = casts.shape[0]
            
                if not drop_others:
                    # add label "others" to casts
                    casts.loc[num_casts] = ['no_exist_others.jpg', 'others']
                    candidates = candidate_json
                else:
                    # remove label "others" in origin candidate_json
                    candidates = candidate_json[candidate_json[0] != "others"]
            
                # self.all_data[mov] = [ candidates, casts ]
                self.all_candidates[mov] = candidates
                self.all_casts[mov] = casts
        
        elif action in ('save', 'val'):
            self.all_candidates = {}
            self.all_casts = {}
            for mov in self.movies:
                # Read json as pandas.DataFrame and divide candidates and others
                candidate_json = pd.read_json(os.path.join(data_path, mov, 'candidate.json'),
                                    orient='index', typ='series').reset_index()
                casts = pd.read_json(os.path.join(data_path, mov, 'cast.json'),
                                    orient='index', typ='series').reset_index()
                
                self.all_candidates[mov] = candidate_json
                self.all_casts[mov] = casts
        # if action in 'test': Do nothing
        elif action in ('test'):
            pass

    def __len__(self):
        return len(self.movies)

    def __getitem__(self, index):
        moviename = self.movies[index]
        if self.action in ('test'):
            # Scanning the folder list
            '''
              Return:
              - images (torch.tensor) : all casts transformed images 
              - moviename (str)       : let candidate dataset can be selected by this mov
              - file_name_list (list) : 
                  all casts img file name (list of str (no ".jpg") )
                  ['tt1840309_0000', 'tt1840309_0001', ...]
            '''
            movie_path  = os.path.join(self.data_path, moviename)
            casts_files = os.listdir(os.path.join(movie_path, 'cast'))

            images = torch.tensor([])
            file_name_list = []

            for cast_file in casts_files:
                image_path = os.path.join(movie_path, 'cast', cast_file)
                image = Image.open(image_path)

                if self.transform:
                    image = self.transform(image)

                images = torch.cat((images,image.unsqueeze(0)), dim=0)
                file_name_list.append(cast_file.split('.')[0])
            
            return images, moviename, file_name_list
        
        elif self.action in ('save', 'val'):
            # cast: all peoples, no others
            # candidates: all images
            
            # Read json as pandas.DataFrame and divide candidates and others
            casts_df      = self.all_casts[moviename]            
            # candidates_df = self.all_candidates[moviename]
            
            # self.casts = casts_df

            num_casts = casts_df.shape[0]

            images = torch.tensor([])
            label_list = []
            img_names = []
            for idx in range(num_casts):
                image_path, label_str = casts_df.iat[idx, 0], casts_df.iat[idx, 1]
                img_name = image_path.split('/')[-1].split('.')[0]
                img_names.append(img_name)

                image = Image.open(os.path.join(self.root_path, image_path))
                if self.transform:
                    image = self.transform(image)
                images = torch.cat((images, image.unsqueeze(0)), dim=0)

                # string label >> int label
                label_mapped = casts_df.index[casts_df[0] == label_str].tolist()
                # handle : if no "others" in cast_df, return self.num_casts as label (old labes : 0 ~ self.num_casts - 1)
                label_mapped = label_mapped[0] if len(label_mapped) > 0 else num_casts
                # label_mapped = torch.tensor(label_mapped)
                label_list.append(label_mapped)
            
            labels = torch.tensor(label_list, dtype=torch.long)   
            return images, labels, moviename, img_names

        elif self.action == 'train':
            # Read json as pandas.DataFrame and divide candidates and others
            # candidates_df = self.all_candidates[moviename]
            casts_df      = self.all_casts[moviename]            

            num_casts = casts_df.shape[0]

            if not self.drop_others:
                # 1. 
                # If don't drop_others, choose 1 images('others') randomly
                # others = candidates_df[candidates_df[0] == "others"]
                # rn = int(torch.randint(0, len(others), (1,)).tolist()[0])
                # casts_df.loc[num_casts] = [others.iat[rn, 0], 'others']
                # num_casts += 1

                # 2. handle when mapping label 
                pass

            images = torch.tensor([])
            label_list = []
            # img_names = []
            for idx in range(num_casts):
                image_path, cast = casts_df.iat[idx, 0], casts_df.iat[idx, 1]
                # img_name = image_path.split('/')[-1].split('.')[0]

                image = Image.open(os.path.join(self.root_path, image_path))
                if self.transform:
                    image = self.transform(image)
                images = torch.cat((images, image.unsqueeze(0)), dim=0)

                # string label >> int label
                label_mapped = casts_df.index[casts_df[0] == cast].tolist()
                # handle : if no "others" in cast_df, return self.num_casts as label (old labes : 0 ~ self.num_casts - 1)
                label_mapped = label_mapped[0] if len(label_mapped) > 0 else num_casts
                # label_mapped = torch.tensor(label_mapped)
                
                label_list.append(label_mapped)
                # img_names.append(img_name)
            labels = torch.tensor(label_list, dtype=torch.long)
                
            return images, labels, moviename    #, img_names

def dataloader_unittest(debug=False):

    print('Save mode testing')

    mode, drop = ('save', False)

    '''
        Warn : 
            Dataset : when action = 'save' , will parse all json files recursicely,
                        so only used when datapath given '<root>/train/' or '<root>/val/'
                        '<root>/test' has to save_features through inference_csv.py
    '''
    for datapath_name in ['train', 'val']:

        print("Testing save '{}'".format(datapath_name))

        cand_dataset = CandDataset(
            data_path = "./IMDb_resize/{}".format(datapath_name),
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]), 
            drop_others=drop,
            action='save',
            load_feature=False
        )

        cast_dataset = CastDataset(
            data_path = "./IMDb_resize/{}".format(datapath_name),
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            drop_others=drop,
            action='save',
            load_feature=False
        )

        cand_loader = DataLoader(cand_dataset, batch_size=8, shuffle=False, num_workers=0)
        cast_loader = DataLoader(cast_dataset, batch_size=1, shuffle=False, num_workers=0)

        print("Length of cast dataset: {}".format(len(cast_dataset)))
        
        # cannot test in this way, because len of cand_dataset is decided by moviename dynamicly
        # print("Length of cand dataset: {}".format(len(cand_dataset)))

        for index, (images, labels, moviename, img_names) in enumerate(cast_loader, 1):
            moviename = moviename[0]    # handle key error, moviename = ('tt1345836' ,) to 'tt1345836'
            cand_dataset.set_mov_name_save(moviename)
            for j, (image, label_mapped, img_name) in enumerate(cand_loader, 1):
                print("cast images.shape: {}".format(images.shape))
                print("cand 0 image.shape: {}".format(image.shape))
                print("cast labels.shape: {}".format(labels.shape))
                print("cand label_mapped.shape: {}".format(label_mapped.shape))
                print("moviename :", moviename)
                print("img_name :", img_name)
                print()
                break
            break

    ########################################################3

    for mode, drop in [('train', True), ('val', False), ('test', False)]:

        print("Dataset setting, action = {}".format(mode))

        cand_dataset = CandDataset(
            data_path = "./IMDb_resize/{}".format(mode),
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]), 
            drop_others=drop,
            action=mode,
            load_feature=False
        )

        cast_dataset = CastDataset(
            data_path = "./IMDb_resize/{}".format(mode),
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            drop_others=drop,
            action=mode,
            load_feature=False
        )

        cand_loader = DataLoader(cand_dataset, batch_size=8, shuffle=False, num_workers=0)
        cast_loader = DataLoader(cast_dataset, batch_size=1, shuffle=False, num_workers=0)

        print("Length of cast dataset: {}".format(len(cast_dataset)))
        
        # cannot test in this way, because len of cand_dataset is decided by moviename dynamicly
        # print("Length of cand dataset: {}".format(len(cand_dataset)))

        if mode == 'train':
            for index, (images, labels, moviename) in enumerate(cast_loader, 1):
                moviename = moviename[0]    # handle key error, moviename = ('tt1345836' ,) to 'tt1345836'
                cand_dataset.set_mov_name_train(moviename)
                for j, (image, label_mapped, index) in enumerate(cand_loader, 1):
                    print("cast images.shape: {}".format(images.shape))
                    print("cand 0 image.shape: {}".format(image.shape))
                    print("cast labels.shape: {}".format(labels.shape))
                    print("cand label_mapped.shape: {}".format(label_mapped.shape))
                    print("moviename :", moviename)
                    print("index :", index)
                    print()
                    break
                break
        elif mode == 'val':
            for index, (images, labels, moviename, img_names) in enumerate(cast_loader, 1):
                moviename = moviename[0]    # handle key error, moviename = ('tt1345836' ,) to 'tt1345836'
                cand_dataset.set_mov_name_val(moviename)
                for j, (image, label_mapped, idx) in enumerate(cand_loader, 1):
                    print("cast images.shape: {}".format(images.shape))
                    print("cand 0 image.shape: {}".format(image.shape))
                    print("cast labels.shape: {}".format(labels.shape))
                    print("cand label_mapped.shape: {}".format(label_mapped.shape))
                    print("moviename :", moviename)
                    print("idx :", idx)
                    print()
                    break
                break       
        elif mode == 'test':
            for index, (images, moviename, file_name_list) in enumerate(cast_loader, 1):
                moviename = moviename[0]    # handle key error, moviename = ('tt1345836' ,) to 'tt1345836'
                cand_dataset.set_mov_name_test(moviename)
                for j, (image, img_name) in enumerate(cand_loader, 1):
                    print("cast images.shape: {}".format(images.shape))
                    print("cand 0 image.shape: {}".format(image.shape))
                    print("moviename :", moviename)
                    print("cast file_name_list :", file_name_list)
                    print("cand img_name :", img_name)
                    print()
                    break
                break

        print("Finish unit testing of mode {}\n\n".format(mode))


if __name__ == "__main__":
    dataloader_unittest()
