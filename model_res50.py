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
      
   feature_extractor_face(VGGFace2 pretrained)---->> model name
   feature_extractor_orugin(Imagenet pretrained)
   classifier--------->>fc_output  dropout
                           drop = 0 ->no dropout
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class feature_extractor_face(nn.Module):
    def __init__(self,model = 'pretrain/resnet50_ft_weight.pkl'):
        super(feature_extractor_face, self).__init__()
        import torchvision
        import pickle
        
        resnet = torchvision.models.resnet50(num_classes=8631,pretrained = False)
        for param in resnet.parameters():
            param.requires_grad = False
        with open(model, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}

        resnet.load_state_dict(weights, strict=False)
        fc_features = resnet.fc.in_features
        resnet.fc = nn.Linear(fc_features, 2048)
        
        self.resnet_layer = nn.Sequential(resnet,
                            )

    def forward(self, input_data):        
        feature = self.resnet_layer(input_data)
        return feature

class feature_extractor_origin(nn.Module):
    def __init__(self):
        super(feature_extractor_origin, self).__init__()
        import torchvision
        resnet = torchvision.models.resnet50(pretrained = True)
        for param in resnet.parameters():
            param.requires_grad = False
        fc_features = resnet.fc.in_features
        resnet.fc = nn.Linear(fc_features, 2048)
        self.resnet_layer = nn.Sequential(resnet,
                            )

    def forward(self, input_data):        
        feature = self.resnet_layer(input_data)
        return feature




class classifier(nn.Module):
    def __init__(self,fc_out = 1024,drop = 0.5):
        super(classifier, self).__init__()
        self.resnet_classifier = nn.Sequential(
                            nn.BatchNorm1d(2048),
                            nn.ReLU(),
                            nn.Dropout(drop),
                            nn.Linear(2048,fc_out),
                            )

    def forward(self, input_data):        
        feature = self.resnet_classifier(input_data)
        return feature

def model_structure_unittest():
    """ Debug model structure """
    
    imgs = Variable(torch.FloatTensor(8, 3, 224, 224))
    res = feature_extractor_face()
    res_last = classifier()
    output = res(imgs)

    feature = res_last(output)
    print(feature.size())


if __name__ == "__main__":
    model_structure_unittest()
