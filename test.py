"""
  FileName     [ test.py ]
  PackageName  [ final ]
  Synopsis     [ Test the Person_reID model with the IMDb dataset ]

  Dataset:
  - IMDb

  Library:
  - apex: A PyTorch Extension, Tools for easy mixed precision and distributed training in Pytorch
          https://github.com/NVIDIA/apex
  - yaml: A human-readable data-serialization language, and commonly used for configuration files.

  Pretrain network:
  - PCB:
  - DenseNet:
  - NAS:
  - ResNet: 
"""

from __future__ import division, print_function

import argparse
import math
import os
import subprocess
import time

import numpy as np
import scipy.io
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import imdb
import utils
import evaluate_rerank
from model import PCB, PCB_test, ft_net, ft_net_dense, ft_net_NAS

# try:
#     from apex.fp16_utils import *
# except ImportError: # will be 3.x series
#     print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

# set gpu / cpu
device = utils.selectDevice()
use_gpu = torch.cuda.is_available()

# ------------------------------------
# Load Data
# ---------
#
# Tranforms functin description:
#   TenCrop(224):
#   Lambda():
# ------------------------------------

if not opt.PCB:
    data_transforms = transforms.Compose([
            transforms.Resize((256,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #transforms.TenCrop(224),
            #transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.ToTensor()(crop) 
            #      for crop in crops]
            # )),
            #transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
            #       for crop in crops]
            # ))
    ])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

image_datasets = imdb.IMDbTrainset(
    movie_path=os.path.join(opt.testset), 
    feature_path=None, 
    label_path=opt.testset+"_GT.json",
    mode='features',
    transform=data_transforms
)
dataloaders = torch.utils.data.DataLoader(
    image_datasets, 
    batch_size=opt.batchsize,
    shuffle=False,
    num_workers=8
)

# class_names = image_datasets['query'].classes

# --------------------------------------
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def extract_feature(model, loader):
    """
      Use Pretrain network to extract the features.

      Params:
      - model: The CNN feature extarctor
      - loader:

      Return:
      - features
    """
    features = torch.FloatTensor()
    
    for index, (img) in enumerate(loader, 1):
        n = img.size()[0]
        
        print('[{:5s}] [Iteration {:4d}/{:4d}]'.format('val', index, len(loader)))
 
        if not opt.PCB:
            ff = torch.FloatTensor(n, 512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, opt.num_part).zero_().cuda() # we have six parts

        # Run the images with normal, horizontal flip
        for i in range(2):
            if i == 1:
                img = utils.fliplr(img)
            
            input_img = img.cuda()
            for scale in ms:
                if scale != 1:
                    # bicubic is only available in pytorch >= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        
        # -----------------------------------------------------------------------------------
        # norm feature
        # feature size (n, 2048, 6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        # ------------------------------------------------------------------------------------    
        if opt.PCB:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        # FF.shape = (n, 2048 * num_part)
        # print("FF.shape: {}".format(ff.shape))

        # Features = (images, 2048 * num_part)
        features = torch.cat((features, ff.data.cpu()), 0)
        # print("Features.shape: {}".format(features.shape))
    
    return features

# (Deprecated 20190614)
def get_id(img_path):
    """
      Get the image information: (camera_id, labels)

      Params:
      - img_path

      Return:
      - camera_id
      - labels
    """
    camera_id = []
    labels = []
    
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    
    return camera_id, labels

def main():
    if not opt.features:
        # ---------------------------------------------------------------------
        # We need to transfrm all candidates images and cast images to features
        # And query the cast inside the same films
        # ---------------------------------------------------------------------
        candidate_paths, candidate_films = image_datasets.candidates['level_1'], image_datasets.candidates['level_0']
        cast_paths, cast_films = image_datasets.casts['level_1'], image_datasets.casts['level_0']

        # -----------------------------------
        # Load datas and trained model
        # -----------------------------------
        if opt.use_dense:
            model_structure = ft_net_dense(opt.nclasses)
        elif opt.use_NAS:
            model_structure = ft_net_NAS(opt.nclasses)
        else:
            model_structure = ft_net(opt.nclasses, stride = opt.stride)

        if opt.PCB:
            model_structure = PCB(opt.nclasses)

        model = utils.load_network(model_structure, opt.resume)

        # Remove the final fc layer and classifier layer
        if opt.PCB:
            model = PCB_test(model)
        else:
            model.classifier.classifier = nn.Sequential()

        # Change to test mode
        model = model.eval()

        # Throught it to the gpu
        if use_gpu:
            model = model.cuda()

        # Extract feature
        with torch.no_grad():
            features = extract_feature(model, dataloaders)

            # candidates first
            num_candidates = dataloaders.dataset.candidates.shape[0]
            
            candidate_feature = features[:num_candidates]
            cast_feature = features[num_candidates:]

        candidate_feature = candidate_feature.numpy()
        candidate_paths   = candidate_paths.to_numpy()
        candidate_films   = candidate_films.to_numpy()
        cast_feature      = cast_feature.numpy()
        cast_paths        = cast_paths.to_numpy()
        cast_films        = cast_films.to_numpy()

        # Save to Matlab for check
        result = {
            'candidate_features': candidate_feature, 
            'candidate_paths': candidate_paths,
            'candidate_films': candidate_films,
            'cast_features': cast_feature, 
            'cast_paths': cast_paths,
            'cast_films': cast_films, 
        }

        print("Features saved to {}".format(os.path.join(opt.output, os.path.basename(opt.resume).split('.')[0] + '_result.mat')))
        scipy.io.savemat('result.mat', result)
        scipy.io.savemat(os.path.join(opt.output, os.path.basename(opt.resume).split('.')[0] + '_result.mat'), result)

    if opt.features:
        result = scipy.io.loadmat(opt.features)

        cast_feature = result['cast_features']
        cast_path = result['cast_paths']
        cast_film = result['cast_films']
        candidate_feature = result['candidate_features']
        candidate_path = result['candidate_paths']
        candidate_film = result['candidate_films']

    print(cast_feature.shape)
    print(cast_film)
    raise NotImplementedError
    re_rank = evaluate_rerank.run(cast_feature, candidate_feature, opt.k1, opt.k2, opt.lambda_value)
    print(re_rank)

    # subprocess.call(['evaluate_gpu.py', *args])

    # result = './model/{}/result.txt'.format(opt.name)
    # os.system('python evaluate_gpu.py | tee -a {}'.format(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing')
    # Device Setting
    parser.add_argument('--gpu_ids', default=[0], nargs='*', type=int, help='gpu_ids: e.g. 0  0 1 2  0 2')
    # Model and dataset Setting
    parser.add_argument('--resume', type=str, help='Directory to the checkpoint')
    parser.add_argument('--testset', default='./IMDb/val', type=str, help='Directory of the validation set')
    parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
    parser.add_argument('--features', type=str, help='Directory of the features.mat')
    # I/O Setting
    parser.add_argument('--output', default='./output', type=str, help='Directory of the output path')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    # Model Setting
    parser.add_argument('--num_part', default=6, type=int, help='A parameter of PCB network.')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
    parser.add_argument('--PCB', action='store_true', help='use PCB' )
    # parser.add_argument('--multi', action='store_true', help='use multiple query' )
    parser.add_argument('--ms', default=[1.0], nargs='*', type=float, help="multiple_scale: e.g. '1' '1 1.1'  '1 1.1 1.2'")
    # Set k-reciprocal Encoding
    parser.add_argument('--k1', default=20, type=int)
    parser.add_argument('--k2', default=6, type=int)
    parser.add_argument('--lambda_value', default=0.3, type=float)

    opt = parser.parse_args()

    # ---------------------------------
    # Load configuration of this model
    # ---------------------------------
    if not opt.features:
        config_path = os.path.join(os.path.dirname(opt.resume), 'opts.yaml')
        with open(config_path, 'r') as stream:
            config = yaml.load(stream)

        opt.name = config['name']
        opt.PCB = config['PCB']
        opt.use_dense = config['use_dense']
        opt.use_NAS = config['use_NAS']
        opt.stride = config['stride']

        if 'nclasses' in config: # tp compatible with old config files
            opt.nclasses = config['nclasses']
        else: 
            opt.nclasses = 199 # Including "others"

    opt.gpu_ids = list(filter(lambda x: x >= 0, opt.gpu_ids))
    ms = [math.sqrt(float(s)) for s in opt.ms]

    # set gpu ids
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        cudnn.benchmark = True
        
    utils.details(opt)
    main()
