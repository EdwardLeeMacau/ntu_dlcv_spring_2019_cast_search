# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 07:55:44 2019

@author: Chun

Usage:
    python main_tri.py --dataroot <IMDb_folder_path> --mpath <model_output_path>
"""
import torch
import argparse
import os
import csv
import numpy as np

from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# from model import feature_extractor
from model_res50 import feature_extractor 
from imdb import TripletDataset, CastDataset
from tri_loss import triplet_loss
from evaluate_rerank import predict_1_movie as predicting
import final_eval

y = {
    'train_loss': [],
    'val_mAP': []
    }

def train(castloader, candloader, cand_data, model, scheduler, optimizer, epoch, device, opt):
    
    scheduler.step()
    model.train()
    
    movie_loss = 0.0
#    print(len(castloader))
    
    for i, (cast, label_cast, mov) in enumerate(castloader):
        mov = mov[0]
#        print('cast size' , cast.size())
#        print(label_cast, type(label_cast))
#            cast_size = 1, num_cast+1, 3, 448, 448
        num_cast = len(label_cast[0])-1
        running_loss = 0.0
        cand_data.mv = mov
        for j, (cand, label_cand, _) in enumerate(candloader):
            
            bs = cand.size()[0]
#            print('candidate size : ', cand.size())
#            print(label_cand, type(label_cand))
#               cand_size = bs - 1 - num_cast, 3, 448, 448
            optimizer.zero_grad()
            
            inputs = torch.cat((cast.squeeze(0),cand), dim=0)
            label = torch.cat((label_cast[0],label_cand), dim=0).tolist()
#            print(label)
            inputs = inputs.to(device)
            
#            print('input size :', inputs.size())  # 16,3,448,448
            
            out = model(inputs)
            loss = triplet_loss(out, label, num_cast)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*bs
            
            if j % opt.log_interval == 0:
                print('Epoch [%d/%d] Movie [%d/%d] Iter [%d] Loss: %.4f'
                      % (epoch, opt.epochs, i, len(castloader),
                         j, running_loss/((j+1)*(bs+1))))  
            if j == 30:                    
                break
        movie_loss += running_loss
#        print(len(castloader))
    return model, movie_loss/len(castloader)
                
            
def val(castloader, candloader, cast_data, cand_data, model, epoch, opt, device):    
    model.eval()
    results = []

    with torch.no_grad():
        for i, (cast, _, mov) in enumerate(castloader):  #label_cast 1*n tensor
            mov = mov[0]
            cast = cast.to(device)
#            cast_size = 1, num_cast+1, 3, 448, 448
            cast_out = model(cast.squeeze(0))
            cast_out = cast_out.detach().cpu().view(-1,2048)
            
            cand_out = torch.tensor([])
            index_out = torch.tensor([], dtype=torch.long)
            cand_data.mv = mov
            for j, (cand, _, index) in enumerate(candloader):
                cand = cand.to(device)
    #               cand_size = bs - 1 - num_cast, 3, 448, 448
                out = model(cand)
                out = out.detach().cpu().view(-1,2048)
                cand_out = torch.cat((cand_out,out), dim=0)
                index_out = torch.cat((index_out, index), dim=0)

                if j == 30:
                    break        

            print('[Validating] ',mov,'processed...',cand_out.size()[0])
            
            cast_feature = cast_out.numpy()
            candidate_feature = cand_out.numpy()
            cast_name = cast_data.casts
            cast_name = np.array([cast_name.iat[x,0][-23:][:-4] 
                                        for x in range(len(cast_name[0]))])
            candidate_name = cand_data.all_data[mov][0]

            candidate_name = np.array([candidate_name.iat[int(index_out[x]),0][-18:][:-4] 
                                        for x in range(cand_out.shape[0])])
#           print(cast_name)
#            print(candidate_name)
            result = predicting(cast_feature, cast_name, candidate_feature, candidate_name)   
            results.extend(result)
    with open('result.csv','w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id','Rank'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    mAP = final_eval.eval('result.csv', os.path.join(opt.dataroot , "val_GT.json"))
#        mAP = cal_map(cast_out, cand_out).cpu()
    return mAP
# --------------------------
# -----  Save model  -------
# --------------------------
def save_network(network, epoch, device, opt, num_fill=3):
    os.makedirs(opt.mpath, mode=0o777, exist_ok=True)
    save_path = os.path.join(opt.mpath, 'net_{}.pth'.format(str(epoch).zfill(num_fill)))
    torch.save(network.cpu().state_dict(), save_path)

    if torch.cuda.is_available():
        network.to(device)

    return


def write_record(record, filename, folder):
    path = os.path.join(folder, filename)
    with open(path, 'a') as file:
        file.write(str(record) + '\n')

# ------------------------------
#    main function
#    ---------------------------------
def main(opt):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    device = torch.device("cuda:0")
    
    transform1 = transforms.Compose([
                        # transforms.Resize((224,224), interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                             ])
    
    train_data = TripletDataset(opt.dataroot, os.path.join(opt.dataroot, 'train'),
                                  mode='classify',
                                  drop_others=True,
                                  transform=transform1,
                                  debug=False)
    train_cand = DataLoader(train_data,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            num_workers=0)
    val_data = TripletDataset(opt.dataroot, os.path.join(opt.dataroot, 'val'),
                                  mode='classify',
                                  drop_others=False,
                                  transform=transform1,
                                  debug=False)
    val_cand = DataLoader(val_data,
                            batch_size=opt.batchsize,
                            shuffle=False,
                            num_workers=0)
    
    train_cast_data = CastDataset(opt.dataroot, os.path.join(opt.dataroot, 'train'),
                                  mode='classify',
                                  drop_others=True,
                                  transform=transform1,
                                  debug=False)
    train_cast = DataLoader(train_cast_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
    val_cast_data = CastDataset(opt.dataroot, os.path.join(opt.dataroot, 'val'),
                                  mode='classify',
                                  drop_others=False,
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
    
    os.makedirs(opt.log_path, mode=0o777, exist_ok=True)

    # testing pre-trained model mAP performance
    epoch = 0
    val_mAP = val(val_cast, val_cand,val_cast_data, val_data, model, epoch, opt, device)
    record = 'Pre-trained Epoch [{}/{}]  Valid_mAP: {:.2%}'.format(epoch, opt.epochs, val_mAP)
    print(record)
    write_record(record, 'val_mAP.txt', opt.log_path)

    best_mAP = 0.0
    for epoch in range(opt.epochs +1):
        model, training_loss = train(train_cast, train_cand, train_data,
                                     model, scheduler, optimizer,
                                     epoch, device, opt)
        record = 'Epoch [%d/%d] TrainingLoss: %.4f' % (epoch, opt.epochs, training_loss)
        print(record)
        write_record(record, 'train_movie_avg_loss.txt', opt.log_path )

        if epoch % opt.save_interval == 0:
            save_network(model, epoch, device, opt)
        
        
        if epoch % 5 == 0:
    
            val_mAP = val(val_cast, val_cand,val_cast_data, val_data, model, epoch, opt, device)
            record = 'Epoch [{}/{}]  Valid_mAP: {:.2%}'.format(epoch, opt.epochs, val_mAP)
            print(record)
            write_record(record, 'val_mAP.txt', opt.log_path)
    
            if val_mAP > best_mAP:
                save_path = os.path.join(opt.mpath, 'net_best.pth')
                torch.save(model.cpu().state_dict(), save_path)
    
                if torch.cuda.is_available():
                    model.to(device)
                val_mAP = best_mAP
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training')
    # Model Setting
    # parser.add_argument('--drop_others', action='store_true', help='if true, the image of type others will be keeped.')
    # parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
    # parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    # parser.add_argument('--img_size', default=[448, 448], type=int, nargs='*')

    # Training setting
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize in training')
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--milestones', default=[10, 20, 30], nargs='*', type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--optimizer', default='ADAM', type=str, help='choose optimizer')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--b1', default=0.9, type=float)
    parser.add_argument('--b2', default=0.999, type=float)
    
    # I/O Setting (important !!!)

    parser.add_argument('--mpath',  default='models', help='folder to output images and model checkpoints')
    parser.add_argument('--log_path',  default='log', help='folder to output loss record')
    parser.add_argument('--dataroot', default='/media/disk1/EdwardLee/dataset/IMDb_Resize/', type=str, help='Directory of dataroot')
    # parser.add_argument('--gt_file', default='/media/disk1/EdwardLee/dataset/IMDb/val_GT.json', type=str, help='Directory of training set.')
    # parser.add_argument('--outdir', default='PCB', type=str, help='output model name')
    # parser.add_argument('--resume', type=str, help='If true, resume training at the checkpoint')
    # parser.add_argument('--trainset', default='/media/disk1/EdwardLee/dataset/IMDb_Resize/train', type=str, help='Directory of training set.')
    # parser.add_argument('--valset', default='/media/disk1/EdwardLee/dataset/IMDb_Resize/val', type=str, help='Directory of validation set')
    
    
    # Device Setting
    parser.add_argument('--gpu', default=0, nargs='*', type=int, help='')
    # parser.add_argument('--threads', default=0, type=int)

    # Others Setting
    parser.add_argument('--debug', action='store_true', help='use debug mode (print shape)' )
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--save_interval', default=5, type=int)
    
    opt = parser.parse_args()
    
    main(opt)
                
                
                
