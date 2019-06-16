"""
  FileName     [ evaluate_gpu.py ]
  PackageName  [ final ]
  Synopsis     [ To generate the query_list by using dot product to measure 
                 the distances between cast and candidates. ]

  Usage:
  - python3 evaluate_gpu.py --name PCB
"""
import csv
import os
from tqdm import tqdm
import argparse

import numpy as np
import scipy.io
import torch

import final_eval

def evaluate(qf, ql, qc, gf, gl, gc, default_juke_label='others'):
    # qf contains 1 image feature
    # print("Qf.shape: ", qf.shape)
    # print("GF.shape: ", gf.shape)
    query = qf.view(-1,1)
    # print(query.shape)
    
    # --------------------------------- # 
    # Dot product                       #
    #   score[i] = np.dot(gf[i], query) #
    # Change to l2 distance             #
    #   l2_dist = 2. - 2 * dot_dist     #  
    # --------------------------------- #
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # --------------------------------- # 
    # Dot product                       #
    #   priority from 1 to -1           #
    # Change to l2 distance             #
    #   priority from 0 to 1            #  
    # --------------------------------- #
    index = np.argsort(score)  # from small to large
    index = index[::-1]        # inverse it

    if (ql is None) or (qc is None) or (gl is None) or (gc is None):
        return index

    # -------------------------------- #
    # Calculating the mAP and CMC here #
    # -------------------------------- #
    # good index
    query_index  = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    # ---------------------------------------------------------------------- # 
    # np.intersect1d(arr1, arr2)                                             # 
    #   Treat the np.array() as a set, find the intersection between 2 sets  # 
    #                                                                        # 
    # np.setdiff1d(arr1, arr2)                                               # 
    #   Set subtraction, return the elements in arr1 but not in arr2         #
    # ---------------------------------------------------------------------- #

    good_index  = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index  = np.append(junk_index2, junk_index1)
    
    # Cumulative match curve = CMC
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return index, CMC_tmp

def run(cast_feature, cast_name, cast_film, candidate_feature, candidate_name, candidate_film, gt, output):
    """
      Run the mAP validation process.

      Params:

      Return:
      - mAP
    """
    cast_feature, candidate_feature = cast_feature.cuda(), candidate_feature.cuda()
    
    result = []
    for i in tqdm(range(cast_feature.shape[0])):
        mask_tensor = torch.from_numpy((candidate_film == cast_film[i]).astype(np.uint8)).byte()
        mask_numpy  = mask_tensor.numpy().astype(bool)
        # print("select_film: ", cast_film[i])
        # print(mask.dtype)
        print(candidate_feature[mask_tensor].shape)
        print(candidate_name[mask_numpy].shape)

        index = evaluate(cast_feature[i], None, None, candidate_feature[mask_tensor], None, None)
        names = candidate_name[mask_numpy][index]
        cast_id = cast_name[i]
        
        # print("Index.shape: ", index.shape)
        # print("Candidates:  ", names.shape)
        # print("Cast_id:     ", cast_id)

        result.append({
            'Id': cast_id, 
            'Rank': ' '.join(names)
        })

    with open(output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Rank'])

        writer.writeheader()
        for r in result:
            writer.writerow(r)

    mAP = eval.eval(output, gt)

    return mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate_GPU')
    parser.add_argument('--features', type=str, help='directory of the features of cast and candidates.')
    parser.add_argument('--multi', action='store_true', help='if true, ranking multi-cast at the same time')
    parser.add_argument('--predict', action='store_true', help='if true, label is un-availables.')
    parser.add_argument('--gt', type=str, help='directory of the gt.json')
    parser.add_argument('--output', type=str, help='directory of the output.csv')

    opt = parser.parse_args()

    # Load the features
    result = scipy.io.loadmat(opt.features)

    cast_feature = torch.FloatTensor(result['cast_features'])
    cast_name = result['cast_names'].reshape(-1)
    cast_film = result['cast_films'].reshape(-1)
    # print(cast_name)
    # print(cast_film)

    candidate_feature = torch.FloatTensor(result['candidate_features'])
    candidate_name = result['candidate_names'].reshape(-1)
    candidate_film = result['candidate_films'].reshape(-1)
    # print(candidate_name)
    # print(candidate_film)

    print(cast_feature.shape)
    print(cast_name.shape)
    print(cast_film.shape)

    print(candidate_feature.shape)
    print(candidate_name.shape)
    print(candidate_film.shape)

    if not opt.multi:
        mAP = run(cast_feature, cast_name, cast_film, candidate_feature, candidate_name, candidate_film, opt.gt, opt.output)
        print("mAP: {:2%}".format(mAP))

    if opt.multi:
        raise NotImplementedError("Not test yet. ")

        # m_result = scipy.io.loadmat('./mat/multi_query_{}.mat'.format(opt.name))
        mquery_feature = torch.FloatTensor(m_result['mquery_f'])
        mquery_cam = m_result['mquery_cam'][0]
        mquery_label = m_result['mquery_label'][0]
        mquery_feature = mquery_feature.cuda()
        # multiple-query
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
    
        for i in range(len(query_label)):
            mquery_index1 = np.argwhere(mquery_label==query_label[i])
            mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
            mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
            mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
            ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])
        CMC = CMC.float()
        CMC = CMC/len(query_label) #average CMC
        print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
