# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 07:55:44 2019

@author: Chun
"""
import torch
import argparse
import os
from torch.optim import lr_scheduler 
import numpy as np
from model import feature_extractor
from imdb import load_candidate, CastDataset
from tri_loss import triplet_loss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from evaluate_rerank import predict_1_movie as predicting

y = {
    'train_loss': [],
    'val_mAP': []
    }

def train(castloader, candloader, model, scheduler, optimizer, epoch, device, opt):
    
    scheduler.step()
    model.train()
    
    movie_loss = 0.0
    
    for i, (cast, label_cast, mov) in enumerate(castloader):
        
        print('cast size' , cast.size())
#            cast_size = 1, num_cast+1, 3, 448, 448
        num_cast = len(label_cast)-1
        
        running_loss = 0.0
        
        for j, (cand, label_cand) in enumerate(candloader[mov]):
            
            bs = cand.size()[0]
            print('candidate size : ', cand.size())
#               cand_size = bs - 1 - num_cast, 3, 448, 448
            optimizer.zero_grad()
            
            inputs = torch.cat((cast.squeeze(0),cand), dim=0)
            label = torch.cat((label_cast, label_cand))
            
            inputs = inputs.to(device)
            
            print('input size :', inputs.size())  # 16,3,448,448
            
            out = model(inputs)
            
            loss = triplet_loss(out, label, num_cast)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*bs
            
            if j % 5 == 0:
                print('Epoch [%d/%d] Movie [%d/%d] Iter [%d/%d] Loss: %.4f'
                      % (epoch, opt.epochs, i, len(castloader),
                         j, len(candloader[mov]), running_loss/(j*bs)))
        
        movie_loss += running_loss/len(candloader[mov])
        
    return model, movie_loss/len(castloader)
                
            
def val(castloader, candloader, model, epoch, opt):
    
    model.eval()
    with torch.no_grad():
        
        for i, (cast, label_cast) in enumerate(castloader):
        
#            cast_size = 1, num_cast+1, 3, 448, 448
            cast_out = model(cast.squeeze(0))
            cast_out = cast_out.detach().cpu().view(-1,2048)
            
            cand_out = torch.tensor([])
            
            for j, (cand, label_cand) in enumerate(candloader):
                
    #               cand_size = bs - 1 - num_cast, 3, 448, 448
                out = model(cand)
                out = out.detach().cpu().view(-1,2048)
                cand_out = torch.cat((cand_out,out), dim=0)
#                labels.extend(label_cand)
                
            print(cand_out.size())
#            print(labels)
            
            cast_feature = cast_out.numpy()
            cast_name = np.array(label_cast)
            candidate_feature = cand_out.numpy()
            candidate_name = np.array(list(range(len(candloader))))
            result = predicting(cast_feature, cast_name, candidate_feature, candidate_name)   
        
#        mAP = cal_map(cast_out, cand_out).cpu()
        
    return 
# --------------------------
# -----  Save model  -------
# --------------------------
def save_network(network, epoch, device, num_fill=3):
    save_path = os.path.join('./model', opt.name, 'net_{}.pth'.format(str(epoch).zfill(num_fill)))
    torch.save(network.cpu().state_dict(), save_path)

    if torch.cuda.is_available():
        network.to(device)

    return

# ------------------------------
#    main function
#    ---------------------------------
    
def main(opt):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    device = torch.device("cuda:0")
    
    transform1 = transforms.Compose([
                        transforms.Resize((448,448), interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                             ])
    
    train_data, train_cand = load_candidate(opt.dataroot,
                                            opt.trainset,
                                            opt.batchsize)
    val_data, val_cand = load_candidate(opt.dataroot,
                                            opt.valset,
                                            opt.batchsize)
    
    train_cast_data = CastDataset(opt.dataroot, opt.trainset,
                                  mode='classify',
                                  keep_others=True,
                                  transform=transform1,
                                  debug=False)
    train_cast = DataLoader(train_cast_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
    val_cast_data = CastDataset(opt.dataroot, opt.trainset,
                                  mode='classify',
                                  keep_others=False,
                                  transform=transform1,
                                  debug=False)
    val_cast = DataLoader(val_cast_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
    
    model = feature_extractor()
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 betas=(opt.b1, opt.b2))  
      
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    
#    since = time.time()
    
    for epoch in range(opt.epochs +1):
        
        model, training_loss = train(train_cast, train_cand,
                                     model, scheduler, optimizer,
                                     epoch, device, opt)
        
        if epoch % opt.save_interval == 0:
            save_network(model, epoch, device)
        
        val_mAP = val(val_cast, val_cand,
                      model, epoch, opt)
        
#        y['val_mAP'].append(val_mAP)

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training')
    # Model Setting
    parser.add_argument('--keep_others', action='store_true', help='if true, the image of type others will be keeped.')
    # parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--img_size', default=[448, 448], type=int, nargs='*')
    # Training setting
    parser.add_argument('--batchsize', default=16, type=int, help='batchsize in training')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--milestones', default=[10, 20, 30], nargs='*', type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str, help='choose optimizer')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--b1', default=0.9, type=float)
    parser.add_argument('--b2', default=0.999, type=float)
    # I/O Setting 
    parser.add_argument('--outdir', default='PCB', type=str, help='output model name')
    parser.add_argument('--mpath',      default='p2_models', help='folder to output images and model checkpoints')
    parser.add_argument('--resume', type=str, help='If true, resume training at the checkpoint')
    parser.add_argument('--dataroot', default='IMDb', type=str, help='Directory of dataroot')
    parser.add_argument('--trainset', default='IMDb/train', type=str, help='Directory of training set.')
    parser.add_argument('--valset', default='IMDb/val', type=str, help='Directory of validation set')
    # Device Setting
    parser.add_argument('--gpu', default=0, nargs='*', type=int, help='')
    parser.add_argument('--threads', default=0, type=int)
    # Others Setting
    parser.add_argument('--debug', action='store_true', help='use debug mode (print shape)' )
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--save_interval', default=1, type=int)
    
    opt = parser.parse_args()
    
    main(opt)
                
                
                