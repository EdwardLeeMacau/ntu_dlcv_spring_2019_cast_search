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
from model import PCB, PCB_test, ft_net, ft_net_dense, ft_net_NAS

# try:
#     from apex.fp16_utils import *
# except ImportError: # will be 3.x series
#     print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default=[0], nargs='*', type=int, help='gpu_ids: e.g. 0  0 1 2  0 2')
parser.add_argument('--resume', type=str, help='Directory to the checkpoint')
# parser.add_argument('--which_epoch', default='last', type=str, help='0, 1, 2, 3...or last')
parser.add_argument('--testset', default='./IMDb/val', type=str, help='Directory of the validation set')
parser.add_argument('--output', default='./output', type=str, help='Directory of the output path')
parser.add_argument('--num_part', default=6, type=int, help='A parameter of PCB network.')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms', default=[1.0], nargs='*', type=float, help="multiple_scale: e.g. '1' '1 1.1'  '1 1.1 1.2'")

opt = parser.parse_args()

# ------------------
# Load configuration
# -------------------
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
    print(config)

opt.fp16 = config['fp16'] 
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

image_datasets = {x: imdb.IMDbTrainset(os.path.join(opt.testset), data_trasnforms) for x in ['candidate', 'cast']}
dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], 
    batch_size=opt.batchsize,
    shuffle=False, num_workers=16) for x in ['candidate','cast']
}

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
    
    for index, (img, _) in enumerate(loader, 1):
        n = img.size()[0]
        print("[{}/{}]".format(index, len(loader)))
        
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, opt.num_part).zero_().cuda() # we have six parts

        # Run the images with normal, horizontal flip
        for i in range(2):
            if i == 1:
                img = utils.fliplr(img)
            
            img = img.cuda()
            for scale in ms:
                if scale != 1:
                    # bicubic is only available in pytorch >= 1.1
                    img = nn.functional.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(img) 
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

        print("FF.shape: {}".format(ff.shape))

        features = torch.cat((features, ff.data.cpu()), 0)
        print("Features.shape: {}".format(features.shape))
    
    return features

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
    candidate_path = image_datasets['candidate'].imgs
    cast_path = image_datasets['cast'].imgs

    gallery_cam, gallery_label = get_id(candidate_path)
    query_cam, query_label = get_id(cast_path)

    # -----------------------------------
    # Load Collected data Trained model
    # -----------------------------------

    if opt.PCB:
        model_structure = PCB(opt.nclasses)

    if opt.use_dense:
        model_structure = ft_net_dense(opt.nclasses)
    elif opt.use_NAS:
        model_structure = ft_net_NAS(opt.nclasses)
    else:
        model_structure = ft_net(opt.nclasses, stride = opt.stride)

    model = utils.load_network(model_structure, opt.resume)

    # Remove the final fc layer and classifier layer
    if opt.PCB:
        model = PCB_test(model)
    else:
        model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    with torch.no_grad():
        candidate_feature = extract_feature(model, dataloaders['candidate'])
        cast_feature = extract_feature(model, dataloaders['cast'])
        
    # Save to Matlab for check
    result = {
        'candidate_f': candidate_feature.numpy(), 
        # 'gallery_label': gallery_label, 
        # 'gallery_cam': gallery_cam, 
        'cast_f': cast_feature.numpy(), 
        # 'query_label': query_label, 
        # 'query_cam': query_cam
    }
    scipy.io.savemat(os.path.join(opt.output, 'pytorch_result.mat'), result)

    print(opt.name)
    result = './model/{}/result.txt'.format(opt.name)

    # subprocess.call(['evaluate_gpu.py', *args])
    os.system('python evaluate_gpu.py | tee -a {}'.format(result))

if __name__ == "__main__":
    utils.details(opt)
    main()
