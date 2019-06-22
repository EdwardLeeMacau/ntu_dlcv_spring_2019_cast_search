# -*- coding: utf-8 -*-
"""
@author: Chun

  FileName     [ train.py ]
  PackageName  [ final ]
  Synopsis     [ Dataloader of IMDb dataset ]

  Usage:
  - python train.py --dataroot <trainset> --mpath <model_output>
  >> Train the model front-to-end 

  - python train.py --features --dataroot <trainset> --mpath <model_output>
  >> Train the classifier (fixed the resnet-50)

  - Other pramameters:
  >> Hyperparameter tuning
"""
import argparse
import csv
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import evaluate
import evaluate_rerank
import final_eval
import utils
from imdb import CandDataset, CastDataset
from model_res50 import Classifier, FeatureExtractorFace
from tri_loss import triplet_loss

y = {
    'train_loss': [],
    'val_loss': [],
    'val_mAP': []
}

newline = '' if sys.platform.startswith('win') else '\n'

def train(castloader: DataLoader, candloader: DataLoader, cand_data, 
          feature_extractor: nn.Module, classifier: nn.Module, scheduler, optimizer, 
          epoch, device, opt, feature_dim=1024) -> (nn.Module, nn.Module, float):   
    """
      Return:
      - feature_extractor
      - classifier
      - train_loss: average with movies
    """
    scheduler.step()

    classifier.train()
    if feature_extractor is not None:
        feature_extractor.train()
    
    movie_loss = 0.0
    
    for i, (cast, label_cast, moviename, _) in enumerate(castloader, 1):
        moviename    = moviename[0]
        label_cast   = label_cast[0]
        num_cast     = len(label_cast)
        running_loss = 0.0

        cand_data.set_mov_name_train(moviename)

        for j, (cand, label_cand, _) in enumerate(candloader, 1):    
            bs = cand.size()[0]                         # cand.shape: batchsize, 3, 224, 224
            optimizer.zero_grad()
            
            inputs = torch.cat((cast.squeeze(0), cand), dim=0)
            label  = torch.cat((label_cast, label_cand), dim=0).tolist()
            inputs = inputs.to(device)
            
            # print('input size :', inputs.size())      # input.shape: batchsize, 3, 224, 224
            
            if feature_extractor is not None:
                out = feature_extractor(inputs)
                out = classifier(out)
            else:
                out = classifier(inputs)

            loss = triplet_loss(out, label, num_cast)   # Size averaged loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * bs
            
            if j % opt.log_interval == 0:
                print('Epoch [%d/%d] Movie [%d/%d] Iter [%d] Loss: %.4f'
                      % (epoch, opt.epochs, i, len(castloader),
                         j, running_loss / (j * bs)))
        
        movie_loss += running_loss

    return feature_extractor, classifier, movie_loss / len(castloader)
                
def val(castloader: DataLoader, candloader: DataLoader, cast_data, cand_data, 
        feature_extractor: nn.Module, classifier: nn.Module, criterion,
        epoch, opt, device, feature_dim=1024) -> (float, float):    
    """
      Return: 
      - mAP:
      - loss:
    """
    classifier.eval()
    if feature_extractor is not None:
        feature_extractor.eval()
    
    movie_loss = 0.0

    results_cosine = []
    results_rerank = []

    with torch.no_grad():
        for i, (cast, label_cast, mov, img_names) in enumerate(castloader, 1):
            mov = mov[0]                        # Un-packing list
            
            cast = cast.to(device)              # cast.shape: 1, num_cast+1, 3, 448, 448
            cast_out = feature_extractor(cast.squeeze(0))
            cast_out = classifier(cast_out)
            cast_out = cast_out.detach().cpu().view(-1, feature_dim)
            
            label_cast = torch.tensor(label_cast).squeeze(0)

            cand_out    = torch.tensor([])
            cand_labels = torch.tensor([], dtype=torch.long)
            index_out   = torch.tensor([], dtype=torch.long)

            cand_data.set_mov_name_val(mov)

            print("[Validating] Number of candidates should be equal to: {}".format(
                len(os.listdir(os.path.join(opt.dataroot, 'val', mov, 'candidates')))))

            for j, (cand, label_cand, index) in enumerate(candloader):
                cand = cand.to(device)          # cand.shape: bs, 3, height, wigth

                if feature_extractor is not None:
                    out = feature_extractor(cand)
                    out = classifier(out)
                else:
                    out = classifier(cand)

                out = out.detach().cpu().view(-1, feature_dim)

                cand_out    = torch.cat((cand_out, out), dim=0)
                cand_labels = torch.cat((cand_labels, label_cand), dim=0)
                index_out   = torch.cat((index_out, index), dim=0)      

            cast_feature = cast_out.to(device)
            candidate_feature = cand_out.to(device)

            # DEBUGS: 
            # print(index_out.sort())

            # Calculate L2 Loss if needed.
            if criterion is not None:
                cand_labels, indices = cand_labels.sort(dim=0)      # Sort the features by the labels
                loss = criterion(candidate_feature[indices], cast_feature[cand_labels]).item()
                movie_loss += loss
                print('[Validating] {}/{} {} processed, get {} features, loss {:.4f}'.format(i, len(castloader), mov, cand_out.size()[0], loss))
            else:
                print('[Validating] {}/{} {} processed, get {} features.'.format(i, len(castloader), mov, cand_out.size()[0]))

            # Getting the labels name
            cast_name = img_names.to_numpy()

            # Getting the labels name from dataframe
            # cast_name = cast_data.casts_df
            # cast_name = cast_name['index'].str[-23:-4].to_numpy()
            
            candidate_df = cand_data.all_candidates[mov]
            candidate_name = candidate_df['index'].str[-18:-4].to_numpy()
            
            result = evaluate.cosine_similarity(cast_feature, cast_name, candidate_feature, candidate_name)
            results_cosine.extend(result)

            # result = evaluate_rerank.predict_1_movie(cast_feature, cast_name, candidate_feature, candidate_name)
            results_rerank.extend(result)

    # Generate the csv with submission format
    with open('result_cosine.csv', 'w', newline=newline) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id','Rank'])
        writer.writeheader()
        for r in results_cosine:
            writer.writerow(r)
    
    with open('result_rerank.csv', 'w', newline=newline) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id','Rank'])
        writer.writeheader()
        for r in results_cosine:
            writer.writerow(r)
    
    # Calculate mAP
    mAP, AP_dict = final_eval.eval('result_cosine.csv', os.path.join(opt.dataroot , "val_GT.json"))
    print('[Cosine] mAP: {:.2%}'.format(mAP))
    
    mAP, AP_dict = final_eval.eval('result_rerank.csv', os.path.join(opt.dataroot , "val_GT.json"))
    print('[Rerank] mAP: {:.2%}'.format(mAP))
    
    # for key, val in AP_dict.items():
    #     record = '[Epoch {}] AP({}): {:.2%}'.format(epoch, key, val)
    #     print(record)
    #     write_record(record, 'val_seperate_AP.txt', opt.log_path)

    return mAP, movie_loss / len(candloader)

def save_network(network: nn.Module, name: str, device, opt):
    """
      Save the models with '<opt.mpath>/<name>'

      Return: None
    """
    save_path = os.path.join(opt.mpath, name)
    torch.save(network.cpu().state_dict(), save_path)

    if torch.cuda.is_available():
        network.to(device)

    return

def draw_graph(epoch, y, opt):
    """
      Draw the training graph
    """
    raise NotImplementedError

def write_record(record, filename: str, folder: str):
    path = os.path.join(folder, filename)
    with open(path, 'a') as textfile:
        textfile.write(str(record) + '\n')

    return

# ------------- #
# main function #
# ------------- #
def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    device = torch.device("cuda")
    
    # ------------------------- # 
    # Dataset initialize        # 
    # ------------------------- #

    if opt.features:
        transform = transforms.ToTensor()
    
    if not opt.features:
        transform = transforms.Compose([
            transforms.Resize((224,224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Candidates Datas    
    train_data = CandDataset(
        data_path=os.path.join(opt.dataroot, 'train'),
        drop_others=True,
        transform=transform,
        action='train'
    )
    
    val_data = CandDataset(
        data_path=os.path.join(opt.dataroot, 'val'),
        drop_others=False,
        transform=transform,
        action='val'
    )

    train_cand = DataLoader(train_data, batch_size=opt.batchsize, shuffle=True, num_workers=opt.threads)
    val_cand   = DataLoader(val_data, batch_size=opt.batchsize, shuffle=False, num_workers=opt.threads)

    # Cast Datas
    train_cast_data = CastDataset(
        data_path=os.path.join(opt.dataroot, 'train'),
        drop_others=True,
        transform=transform,
        action='train'
    )

    val_cast_data = CastDataset(
        data_path=os.path.join(opt.dataroot, 'val'),
        drop_others=False,
        transform=transform,
        action='val'
    )

    train_cast = DataLoader(train_cast_data, batch_size=1, shuffle=False, num_workers=opt.threads)
    val_cast   = DataLoader(val_cast_data, batch_size=1, shuffle=False, num_workers=opt.threads)
    
    # ------------------------- # 
    # Model, optim initialize   # 
    # ------------------------- #
    classifier = Classifier(2048).to(device)
    feature_extractor = None
    params = [{'params': classifier.parameters()}]
    if not opt.features:
        feature_extractor = FeatureExtractorFace().to(device)
        params.append({'params': classifier.parameters(), 'lr': 1e-3})
    
    optimizer = torch.optim.Adam(params,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        betas=(opt.b1, opt.b2)
    )
      
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    criterion = nn.MSELoss(reduction='sum') # For validation only.

    # ----------------------------------------- #
    # Testing pre-trained model mAP performance #
    # ----------------------------------------- #
    val_mAP, val_loss = val(val_cast, val_cand, val_cast_data, val_data,
                            feature_extractor, classifier, criterion,
                            0, opt, device, feature_dim=opt.feature_dim)
    record = 'Pre-trained Epoch [{}/{}]  Valid_mAP: {:.2%} Valid_loss: {:.4f}\n'.format(0, opt.epochs, val_mAP, val_loss)
    print(record)
    write_record(record, 'val_mAP.txt', opt.log_path)

    # ----------------------------------------- #
    # Training                                  #
    # ----------------------------------------- #
    best_mAP = 0.0
    for epoch in range(1, opt.epochs + 1):
        # Dynamic adjust the loss margin
        pass

        # Train the models
        # If train classifier only, the variable 'feature_extractor' is set as None
        feature_extractor, classifier, training_loss = train(train_cast, train_cand, train_data,
                                                            feature_extractor, classifier, scheduler, optimizer,
                                                            epoch, device, opt, feature_dim=opt.feature_dim)
            
        y['train_loss'].append(training_loss)

        # Print and log the training loss
        record = 'Epoch [%d/%d] TrainingLoss: %.4f' % (epoch, opt.epochs, training_loss)
        print(record)
        write_record(record, 'train_movie_avg_loss.txt', opt.log_path )

        # Save the network
        if epoch % opt.save_interval == 0:
            name = 'net_{}.pth'.format(str(epoch).zfill(3))
            save_network(classifier, name, device, opt)
        
        # Validate the model performatnce
        if epoch % opt.save_interval == 0:    
            # If train classifier only, the variable 'feature_extractor' is set as None
            val_mAP, val_loss = val(val_cast, val_cand, val_cast_data, val_data, 
                                    feature_extractor, classifier, criterion,  
                                    epoch, opt, device, feature_dim=opt.feature_dim)

            y['val_loss'].append(val_loss)
            y['val_mAP'].append(val_mAP)

            # Print and log the validation loss
            record = 'Epoch [{}/{}] Valid_mAP: {:.2%} Valid_loss: {:.4f}\n'.format(epoch, opt.epochs, val_mAP, val_loss)
            print(record)
            write_record(record, 'val_mAP.txt', opt.log_path)
    
            # Save the best model
            if val_mAP > best_mAP:
                save_network(classifier, 'net_best.pth', device, opt)
                val_mAP = best_mAP
        
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Training')
    
    # Training setting
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize in training')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--milestones', default=[10, 20, 30], nargs='*', type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--b1', default=0.9, type=float)
    parser.add_argument('--b2', default=0.999, type=float)
    parser.add_argument('--feature_dim', default=1024, type=int)
    
    # I/O Setting (important !!!)
    parser.add_argument('--mpath',  default='models', help='folder to output images and model checkpoints')
    parser.add_argument('--log_path', default='log', help='folder to output logs')
    parser.add_argument('--dataroot', default='./IMDb_Resize/', type=str, help='Directory of dataroot')
    parser.add_argument('--features', action='store_true', help='If true, dataloader will load the image in features')
    # parser.add_argument('--gt_file', default='./IMDb_Resize/val_GT.json', type=str, help='Directory of training set.')
    # parser.add_argument('--resume', type=str, help='If true, resume training at the checkpoint')
    
    # Device Setting
    parser.add_argument('--gpu', default=0, nargs='*', type=int, help='')
    parser.add_argument('--threads', default=0, type=int)

    # Others Setting
    # parser.add_argument('--debug', action='store_true', help='use debug mode (print shape)' )
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--save_interval', default=1, type=int, help='Validation and save the network')

    opt = parser.parse_args()

    # Make directories
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.mpath, exist_ok=True)

    # Show the settings, and start to train
    utils.details(opt)
    main(opt)
