# -*- coding: utf-8 -*-
"""
  FileName     [ inference_csv.py ]
  PackageName  [ final ]
  Synopsis     [ To inference trained model with testing images, output csv file ]

  Example:
  - python3 inference_csv.py --dataroot ./IMDb_resize/ --model ./net_best.pth --action test --out_csv ./result.csv --save_feature
  >> 

"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import evaluate_rerank
import evaluate
import final_eval
import utils
from imdb import CandDataset, CastDataset
from model_res50 import (Classifier, FeatureExtractorFace,
                         FeatureExtractorOrigin)

newline = '' if sys.platform.startswith('win') else '\n'

def test(castloader: DataLoader, candloader: DataLoader, cast_data, cand_data, 
         feature_extractor: nn.Module, classifier: nn.Module, 
         opt, device, feature_dim=1024):
    '''
      Inference by trained model, generated inferenced result if needed.

      Return: None
    '''
    print('Start Inferencing {}ing dataset ... '.format(opt.action))    

    classifier.eval()
    if feature_extractor is not None:
        feature_extractor.eval()
    
    results_cosine = []
    results_rerank = []
    
    if opt.save_feature:
        os.makedirs('./feature_np/', exist_ok=True)
        os.makedirs('./feature_np/val/', exist_ok=True)
        os.makedirs('./feature_np/test/', exist_ok=True)

    # ------------------------- # 
    # If ground truth exists    # 
    # ------------------------- #
    if opt.action == 'val':
        for i, (cast, _, mov) in enumerate(castloader):
            mov = mov[0]
            cast = cast.to(device)
            # cast_size = 1, num_cast+1, 3, 448, 448
            cast_out = feature_extractor(cast.squeeze(0))
            cast_out = classifier(cast_out)
            cast_out = cast_out.detach().cpu().view(-1, feature_dim)
            
            cand_out = torch.tensor([])
            index_out = torch.tensor([], dtype=torch.long)
            
            cand_data.mv = mov
            for j, (cand, _, index) in enumerate(candloader):
                cand = cand.to(device)
                # cand_size = bs - 1 - num_cast, 3, w, c
                
                if feature_extractor is not None:
                    out = feature_extractor(cand)
                    out = classifier(out)
                    out = out.detach().cpu().view(-1, 1024)
                else:
                    out = classifier(cand)

                cand_out  = torch.cat((cand_out, out), dim=0)
                index_out = torch.cat((index_out, index), dim=0)       

            print(index_out.sort(dim=0))
            print('[Testing]', mov, 'processed ...', cand_out.size()[0])
            
            casts_features = cast_out# .numpy()
            candidates_features = cand_out# .numpy()

            cast_name = cast_data.casts_df[0].str[-23:-4]
            print(cast_name)
            
            candidate_name = cand_data.all_candidates[mov][0].str[-18:-4]
            print(candidate_name)

            result = evaluate.cosine_similarity(casts_features, cast_name, candidate_features, candidate_name)
            result = evaluate_rerank.predict_1_movie(casts_features, cast_name, candidates_features, candidate_name, 
                                        k1=opt.k1, k2=opt.k2, lambda_value=0.3)
            results.extend(result)

        with open(opt.out_csv, 'w', newline=newline) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Rank'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        
        # mAP, AP_dict = final_eval.eval(opt.out_csv, opt.gt)
        mAP, AP_dict = final_eval.eval(opt.out_csv, os.path.join(opt.dataroot, "val_GT.json"))
        for key, val in AP_dict.items():
            record = 'AP({}): {:.2%}'.format(key, val)
            print(record)
        
        print('[ mAP = {:.2%} ]\n'.format(mAP))

        return

    # --------------------------------- # 
    # If ground truth doesn't exists    # 
    # --------------------------------- #
    if opt.action == 'test':
        for i, (cast, mov, cast_file_name_list) in enumerate(castloader):
            mov = mov[0]    # unpacked from batch

            # [('tt0121765_nm0000204',), ('tt0121765_nm0000168',), ...] to ['tt0121765_nm0000204', 'tt0121765_nm0000168', ...]
            cast_file_name_list = [x[0] for x in cast_file_name_list]

            if not opt.load_feature:
                print("generating {}'s cast features".format(mov))
                cast = cast.to(device)          # cast.size[1, num_cast, 3, 224, 224]
                cast_out = Feature_extractor(cast.squeeze(0))
                cast_out = Feature_extractor_fc(cast_out)
                cast_out = cast_out.detach().cpu().view(-1, 1024)
                casts_features = cast_out.numpy()

                # Save cast features 
                if opt.save_feature:
                    feature_path = './feature_np/test/{}/cast/'.format(mov)
                    os.makedirs(feature_path, exist_ok=True)
                    np.save(os.path.join(feature_path, "features.npy"), casts_features)
                    np.save(os.path.join(feature_path, "names.npy"), cast_file_name_list[0])

                print("generating {}'s candidate features".format(mov))
                cand_data.set_mov_name_test(mov)
                cand_out = torch.tensor([])
                cand_file_name_list = []
                
                for j, (cand, cand_file_name_tuple) in enumerate(candloader):
                    cand_file_name_list.extend(list(cand_file_name_tuple))
                    # count += len(cand_file_name_tuple)
                    # print('count = {}'.format(count))
                    
                    cand = cand.to(device)
                    out = Feature_extractor(cand)
                    out = Feature_extractor_fc(out)
                    out = out.detach().cpu().view(-1, 1024)
                    cand_out = torch.cat((cand_out, out), dim=0)

                candidates_features = cand_out.numpy()

                # Save candidates features
                if opt.save_feature:
                    feature_path = './feature_np/test/{}/candidates/'.format(mov)
                    os.makedirs(feature_path, exist_ok=True)
                    np.save(os.path.join(feature_path, "features.npy"), candidates_features)
                    np.save(os.path.join(feature_path, "names.npy"), cand_file_name_list)

                print('imgs_num({}) / file_names({})'.format(cand_out.size()[0], len(cand_file_name_list)))
            
            else:   # if load_feature:
                print("loading {}'s cast features".format(mov))
                feature_path = './feature_np/test/{}/cast/features.npy'.format(mov)
                names_path = './feature_np/test/{}/cast/names.npy'.format(mov)
                casts_features = np.load(feature_path)
                cast_file_name_list = list(np.load(names_path))

                print("loading {}'s candidate features".format(mov))
                feature_path = './feature_np/test/{}/candidates/features.npy'.format(mov)
                names_path = './feature_np/test/{}/candidates/names.npy'.format(mov)
                candidates_features = np.load(feature_path)
                cand_file_name_list = list(np.load(names_path))

                print('file_names({})'.format(len(cand_file_name_list)))

            print('[Testing] {} processing predict_ranking ... \n'.format(mov))
            
            # predict_ranking
            result = evaluate_rerank.predict_1_movie(casts_features, np.array(cast_file_name_list), candidates_features, np.array(cand_file_name_list),
                                        k1=opt.k1, k2=opt.k2, lambda_value=0.3)
            results.extend(result)

        print('Start writing output csv file')
        with open(opt.out_csv, 'w', newline=newline) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Id','Rank'])
            writer.writeheader()
            
            for r in results:
                writer.writerow(r)
    
        print('Testing output "{}" writed. \n'.format(opt.out_csv))

        return

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    device = torch.device("cuda")

    folder_name = opt.action

    # ------------------------- # 
    # Dataset initialize        # 
    # ------------------------- #
    if not opt.features:
        transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if opt.features:
        transform = transforms.ToTensor()
    
    # Candidate Dataset and DataLOader
    test_data = CandDataset(
        data_path=os.path.join(opt.dataroot, folder_name),
        drop_others=False,
        transform=transfrom,
        action=opt.action
    )
        
    test_cast_data = CastDataset(
        data_path=os.path.join(opt.dataroot, folder_name),
        drop_others=False,
        transform=transform,
        action=opt.action
    )

    test_cand = DataLoader(test_data, batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)
    test_cast = DataLoader(test_cast_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    # ------------------------- # 
    # Model initialize          # 
    # ------------------------- #
    feature_extractor = FeatureExtractorFace().to(device)
    classifier = Classifier().to(device)

    if opt.model:
        print("Parameter read: {}".format(opt.model))
        classifier = utils.load_network(classifier, opt.model).to(device)

    # ------------------------- # 
    # Execute Test Function     # 
    # ------------------------- #
    with torch.no_grad():
        test(test_cast, test_cand, test_cast_data, test_data, feature_extractor, classifier, opt, device)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Testing')
    # Dataset setting
    parser.add_argument('--batchsize', default=128, type=int, help='batchsize in testing (one movie folder each time) ')
    # I/O Setting (important !!!)
    parser.add_argument('--model', help='model checkpoint path to extract features')    # ./model_face/net_best.pth
    parser.add_argument('--features', action='store_true')
    parser.add_argument('--dataroot', default='./IMDb_Resize/', type=str, help='Directory of dataroot')
    parser.add_argument('--action', default='test', type=str, help='action type (test / val)')
    parser.add_argument('--gt', type=str, help='if gt_file is exists, measure the mAP.')
    parser.add_argument('--out_csv',  default='./result.csv', help='output csv file name')
    parser.add_argument('--save_feature', action='store_true', help='save new np features when processing')
    parser.add_argument('--load_feature', action='store_true', help='load old np features when processing')
    # Device Setting
    parser.add_argument('--gpu', default=0, nargs='*', type=int, help='')
    parser.add_argument('--num_workers', default=0, type=int, help='')
    # Others Setting
    parser.add_argument('--debug', action='store_true', help='use debug mode (print shape)' )
    # Rerank Setting
    parser.add_argument('--k1', default=20, type=int, help='')
    parser.add_argument('--k2', default=6, type=int, help='')

    opt = parser.parse_args()
    
    # Prints the setting
    utils.details(opt)

    # Check files here
    if not os.path.exists(opt.dataroot):
        raise IOError("{} is not exists".format(opt.dataroot))
    
    if opt.save_feature and opt.load_feature:
        raise ValueError('Cannot save and load features simultanesouly, please choose one of them')
    
    if opt.model is not None:
        if not os.path.exists(opt.model):
            raise IOError("{} is not exists".format(opt.model))
    
    # if not os.path.exists(opt.gt):
    #     pass
    #     raise IOError("{} is not exists".format(opt.gt))

    # Execute the main function
    main(opt)
