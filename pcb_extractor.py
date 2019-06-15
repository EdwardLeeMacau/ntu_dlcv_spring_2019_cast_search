"""
  FileName     [ pcb_extractor.py ]
  PackageName  [ final ]
  Synopsis     [ Feature extracting method of PCB model. ]

  Dataset:
  - IMDb:

  Model:
  - PCB:
"""

from __future__ import division, print_function

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils

def extract_feature(model: nn.Module, loader: DataLoader, use_gpu=torch.cuda.is_available(), 
                    ms=[1.0], pcb=6) -> torch.Tensor:
    """
      Use Pretrain network to extract the features.

      Params:
      - pcb: 
        PCB larger than 0 means using pcb, and num_part equals the value of pcb
      - ms:  
        Scailing factors in list.

      Return:
      - features:
        The features loaded from dataloader
    """
    features = torch.FloatTensor()
    
    for index, (img) in enumerate(loader, 1):
        n = img.size()[0]
        
        print('[{}] [Iteration {:4d}/{:4d}]'.format('Extracting Feature', index, len(loader)))
 
        if not pcb:
            ff = torch.FloatTensor(n, 512).zero_().cuda()
        if pcb:
            ff = torch.FloatTensor(n, 2048, pcb).zero_().cuda()
            
        # Run the images with normal, horizontal flip
        for i in range(2):
            if i == 1: img = utils.fliplr(img)
            
            input_img = img.cuda()
            for scale in ms:
                if scale != 1:
                    # bicubic is only available in pytorch >= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        
        # -----------------------------------------------------------------------------------
        # norm features:
        #   if opt.PCB:
        #     feature size (n, 2048 x num_part)
        #   if not opt.PCB:
        #     feature size (n, 512)
        # 
        # Notes:
        #   1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        #   2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        # ------------------------------------------------------------------------------------    
        if pcb:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        
        if not pcb:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    
    return features

if __name__ == "__main__":
    raise NotImplementedError("{} is for import only.".format(__file__))