"""
  FileName     [ model.py ]
  PackageName  [ layumi/Person_reID_baseline_pytorch ]
  Synopsis     [ Model class ]

  Library:
  - apex: A PyTorch Extension, Tools for easy mixed precision and distributed training in Pytorch
          https://github.com/NVIDIA/apex
  - yaml: A human-readable data-serialization language, and commonly used for configuration files.
  - pretrainedmodels: 
          Install: pip install pretrainedmodels

  Pretrain network:
  - PCB: Part-based Convolutional Baseline
         https://arxiv.org/abs/1711.09349
         Beyond Part Models: Person Retrieval with Refined Part Pooling (and a Strong Convolutional Baseline)
  - DenseNet:
  - NAS:
  - ResNet: 
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        import torchvision
        resnet = torchvision.models.resnet50(num_classes=8631,pretrained = False)
        import pickle
        with open('pretrain/resnet50_ft_weight.pkl', 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}

        resnet.load_state_dict(weights, strict=False)

        self.resnet_layer = nn.Sequential(*list(resnet.children())[:-2],
                            nn.AdaptiveAvgPool2d(output_size=(1, 1)) 
                            )

    def forward(self, input_data):
        feature = self.resnet_layer(input_data)
        return feature

def model_structure_unittest():
    """ Debug model structure """

    imgs = Variable(torch.FloatTensor(8, 3, 224, 224))
    res = feature_extractor()
    output = res(imgs)
    print(output.shape)
    print(output[0][0])
    # print(res)

if __name__ == "__main__":
    model_structure_unittest()
