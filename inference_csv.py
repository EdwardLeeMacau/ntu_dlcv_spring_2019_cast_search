# -*- coding: utf-8 -*-
"""
FileName     [ inference_csv.py ]
PackageName  [ final ]
Synopsis     [ To inference trained model with testing images, output csv file ]

Usage:
    python inference_csv.py
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
from evaluate_rerank import predict_1_movie as predict_ranking
import final_eval
import utils
              

def test(castloader, candloader, cast_data, cand_data, model, opt, device):    
    model.eval()
    results = []
    action = opt.action

    if action == 'val':
        with torch.no_grad():
            for i, (cast, _, mov) in enumerate(castloader):  #label_cast 1*n tensor
                mov = mov[0]
                cast = cast.to(device)
                # cast_size = 1, num_cast+1, 3, 448, 448
                cast_out = model(cast.squeeze(0))
                cast_out = cast_out.detach().cpu().view(-1,2048)
                
                cand_out = torch.tensor([])
                index_out = torch.tensor([], dtype=torch.long)
                cand_data.mv = mov
                for j, (cand, _, index) in enumerate(candloader):
                    cand = cand.to(device)
                    #    cand_size = bs - 1 - num_cast, 3, 448, 448
                    out = model(cand)
                    out = out.detach().cpu().view(-1,2048)
                    cand_out = torch.cat((cand_out,out), dim=0)
                    index_out = torch.cat((index_out, index), dim=0)       

                print('[Testing]', mov, 'processed ...', cand_out.size()[0])
                
                cast_feature = cast_out.numpy()
                candidate_feature = cand_out.numpy()
                cast_name = cast_data.casts
                cast_name = np.array([cast_name.iat[x,0][-23:][:-4] 
                                            for x in range(len(cast_name[0]))])
                candidate_name = cand_data.all_candidates[mov]
                # candidate_name = cand_data.all_data[mov][0]
                candidate_name = np.array([candidate_name.iat[int(index_out[x]),0][-18:][:-4] 
                                            for x in range(cand_out.shape[0])])
                # print(cast_name)
                # print(candidate_name)
                result = predict_ranking(cast_feature, cast_name, candidate_feature, candidate_name)   
                results.extend(result)

        with open(opt.out_csv,'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Id','Rank'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        
        # -----------------------------
        # when testing val_set
        # -----------------------------
        mAP, AP_dict = final_eval.eval(opt.out_csv, os.path.join(opt.dataroot , "val_GT.json"))
        for key, val in AP_dict.items():
            record = 'AP({}): {:.2%}'.format(key, val)
            print(record)
        print('[ mAP = {:.2%} ]\n'.format(mAP))

    elif action == 'test':
        with torch.no_grad():
            for i, (cast, mov, cast_file_name_list) in enumerate(castloader):  #label_cast 1*n tensor
                mov = mov[0]    # unpacked from batch
                cand_data.mv = mov
                
                # generate cast features
                # -----------------------------
                cast = cast.to(device)          # cast.size[1, num_cast, 3, 224, 224]
                cast_out = model(cast.squeeze(0))
                cast_out = cast_out.detach().cpu().view(-1,2048)
                cast_feature = cast_out.numpy()
                
                # generate candidiate features
                # only 1 movie's candidates would be load in each i (decided by "cand_data.mv")
                for j, (cand, cand_file_name_list) in enumerate(candloader):
                    cand = cand.to(device)
                    cand_out = model(cand.squeeze(0))
                    cand_out = cand_out.detach().cpu().view(-1,2048)
                    candidate_feature = cand_out.numpy()

                print('[Testing]', mov, 'processed ...', cand_out.size()[0])
                
                # predict_ranking
                result = predict_ranking(cast_feature, cast_file_name_list, candidate_feature, cand_file_name_list)   
                results.extend(result)

        with open(opt.out_csv,'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Id','Rank'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)


# ------------------------------
#    main function
# ---------------------------------
def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    device = torch.device("cuda:0")
    if opt.action == 'test':
        folder_name = 'test_resize'
    elif opt.action == 'val':
        folder_name = 'val'
    
    transform1 = transforms.Compose([
                        # transforms.Resize((224,224), interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                             ])
    
    test_data = TripletDataset(opt.dataroot, os.path.join(opt.dataroot, folder_name),
                                  mode='classify',
                                  drop_others=False,
                                  transform=transform1,
                                  debug=opt.debug,
                                  action=opt.action)
    test_cand = DataLoader(test_data,
                            batch_size=opt.batchsize,
                            shuffle=False,
                            num_workers=0)
    
    test_cast_data = CastDataset(opt.dataroot, os.path.join(opt.dataroot, folder_name),
                                  mode='classify',
                                  drop_others=False,
                                  transform=transform1,
                                  debug=opt.debug,
                                  action=opt.action)
    test_cast = DataLoader(test_cast_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
    
    model = feature_extractor()
    model = model.to(device)

    # testing trained model, output result.csv
    test(test_cast, test_cand, test_cast_data, test_data, model, opt, device)

    print('Testing output "{}" writed. \n'.format(opt.out_csv))
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Testing')
    # Dataset setting
    parser.add_argument('--batchsize', default=1, type=int, help='batchsize in training')
    # parser.add_argument('--img_size', default=[448, 448], type=int, nargs='*')
    
    # I/O Setting (important !!!)
    parser.add_argument('--model',  default='./model_face/net_best.pth', help='model checkpoint path to extract features')
    parser.add_argument('--dataroot', default='/media/disk1/EdwardLee/dataset/IMDb_Resize/', type=str, help='Directory of dataroot')
    parser.add_argument('--action', default='test', type=str, help='action type (test / val)')
    parser.add_argument('--out_csv',  default='./result.csv', help='output csv file name')
    # Device Setting
    parser.add_argument('--gpu', default=0, nargs='*', type=int, help='')
    # Others Setting
    parser.add_argument('--debug', action='store_true', help='use debug mode (print shape)' )
    opt = parser.parse_args()
    
    utils.details(opt)
    main(opt)