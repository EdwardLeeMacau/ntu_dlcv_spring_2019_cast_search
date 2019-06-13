"""
  FileName     [ evaluate_rerank.py ]
  PackageName  [ final ]
  Synopsis     [ To evaluate the model performance ]
"""

import csv
import os
import time

import numpy as np
import pandas as pd
import scipy.io
import torch

from re_ranking import re_ranking


# This evaluate function is different with other evaluate functions
def evaluate(score, ql, qc, gl, gc):
    """
      Params:
      - score
      - ql: query_label = result['query_label'][0]
      - qc: query_cam = result['query_cam'][0]
      - gl: gallery_cam = result['gallery_cam'][0]
      - gc: gallery_label = result['gallery_label'][0]
    
      Return:
      - CMC_tmp
    """
    index = np.argsort(score)  #from small to large
    # index = index[::-1]
    
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

# Deprecated (20190614)
def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

def output(path):
    """
      Params:
      - path: the directory to output csv file

      Return: None
    """
    raise NotImplementedError

def run(query_features, gallery_features, k1=20, k2=6, lambda_value=0.3):
    # ------------------------
    # Mapping:
    #   cast -> query
    #   candidate -> gallery
    # ------------------------
    # re-ranking
    print('calculate initial distance')
    q_g_dist = np.dot(query_features, np.transpose(gallery_features))
    q_q_dist = np.dot(query_features, np.transpose(query_features))
    g_g_dist = np.dot(gallery_features, np.transpose(gallery_features))
    
    print('reranking')
    re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=k1, k2=k2, lambda_value=lambda_value)

    return re_rank

def main():
    result_path = os.path.join('./output', 'PCB', 'pytorch_result.mat')

    result = scipy.io.loadmat(result_path)

    query_feature = result['query_f']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_f']
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    # re-ranking
    print('calculate initial distance')
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
    
    print('reranking')
    lambda_value, k1, k2 = 0.3, 20, 6
    re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=k1, k2=k2, lambda_value=lambda_value)
    
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(re_rank[i,:], query_label[i], query_cam[i], gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC / len(query_label) #average CMC
    print('top1: {} top5: {} top10: {} mAP:{}'.format(CMC[0], CMC[4], CMC[9], ap/len(query_label)))

if __name__ == '__main__':
    main()
