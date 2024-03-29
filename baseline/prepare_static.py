"""
  FileName     [ prepare_static.py ]
  PackageName  [ layumi/Person_reID_baseline_pytorch ]
  Synopsis     [ Show the statistics message of the dataset. ]
"""

from __future__ import print_function, division

import argparse
import torch
from torchvision import datasets, transforms
import time
import os

# version =  torch.__version__

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir',default='/home/zzd/Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
opt = parser.parse_args()

def prepare_model():
    """
      To check how many times is needed to train for 1 epoch.
    """
    data_dir = opt.data_dir

    ######################################################################
    transform_train_list = [
            #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((288,144), interpolation=3),
            #transforms.RandomCrop((256,128)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    transform_val_list = [
            transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]


    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }


    train_all = ''
    if opt.train_all:
        train_all = '_all'

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                            data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                            data_transforms['val'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=True, num_workers=16)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    use_gpu = torch.cuda.is_available()

    since = time.time()

    for phase in ['train', 'val']:
        mean = torch.zeros(3)
        std = torch.zeros(3)
        # Iterate over data.
        for (inputs, labels) in dataloaders[phase]:
            now_batch_size,c,h,w = inputs.shape
            mean += torch.sum(torch.mean(torch.mean(inputs,dim=3),dim=2),dim=0)
            std += torch.sum(torch.std(inputs.view(now_batch_size,c,h*w),dim=2),dim=0)
            
        print("{}.mean: {}".format(phase, mean/dataset_sizes[phase]))
        print("{}.std:  {}".format(phase, std/dataset_sizes[phase]))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return 

if __name__ == "__main__":
    prepare_model()
