"""
  FileName     [ model.py ]
  PackageName  [ final ]
  Synopsis     [ Model class ]
      
  1. Feature_extractor_face   (VGGFace2 pretrained) ---->> model name
  2. Feature_extractor_origin (Imagenet pretrained)
  3. Classifier --------->> fc_output  dropout
                            drop = 0 ->no dropout
"""
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

class Feature_extractor_face(nn.Module):
    def __init__(self, model='pretrain/resnet50_ft_weight.pkl'):
        super(Feature_extractor_face, self).__init__()
        
        # load pytorch resnet50
        resnet = torchvision.models.resnet50(num_classes=8631, pretrained=False)

        # fix layers
        for param in resnet.parameters():
            param.requires_grad = False
        
        # load pretrained weights
        with open(model, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        resnet.load_state_dict(weights, strict=False)

        # change output dim (from 8631 to 2048)
        fc_features = resnet.fc.in_features
        resnet.fc = nn.Linear(fc_features, 2048)
        
        # defined all resnet layers
        self.resnet_layer = resnet

    def forward(self, input_data):        
        feature = self.resnet_layer(input_data)
        return feature

class Feature_extractor_origin(nn.Module):
    def __init__(self):
        super(Feature_extractor_origin, self).__init__()

        # load pytorch pretrained resnet50
        resnet = torchvision.models.resnet50(pretrained = True)

        # fix layers
        for param in resnet.parameters():
            param.requires_grad = False

        # change output dim (from 8631 to 2048)
        fc_features = resnet.fc.in_features
        resnet.fc = nn.Linear(fc_features, 2048)

        # defined all resnet layers
        self.resnet_layer = resnet

    def forward(self, input_data):        
        feature = self.resnet_layer(input_data)
        return feature

class Classifier(nn.Module):
    def __init__(self,fc_out = 1024,drop = 0.5):
        super(Classifier, self).__init__()

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
    # img
    imgs = Variable(torch.FloatTensor(8, 3, 224, 224))

    # model
    res = Feature_extractor_face()
    res_last = Classifier()

    # infernece
    output = res(imgs)

    feature = res_last(output)
    print(feature.size())

if __name__ == "__main__":
    model_structure_unittest()
