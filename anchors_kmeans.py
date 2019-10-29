# -*- coding: utf-8 -*-
# @Time    : 2019-10-15 16:40
# @Author  : Yan An
# @Contact: an.yan@intellicold.ai


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import write_dict
from sklearn.cluster import KMeans


def main(args):
    fd_label = args.data_path
    
    hout = args.input_size
    wout = args.input_size
    wh = np.array([])
 
    for root in os.listdir(fd_label):
        fn_label = os.path.join(fd_label,root)
        f,ext = os.path.splitext(fn_label)
        if ext==".txt":
            a = np.loadtxt(fn_label)
            if wh.shape[0]==0:
                wh = a
                if len(wh.shape)==1:
                    wh = np.expand_dims(wh,axis = 0)
            else:
                if len(a.shape)==1:
                    a = np.expand_dims(a,axis = 0)
                wh = np.concatenate((wh,a),axis=0)
    
    wh0 = wh[:, 3:]
    wh0[:,0] = wh0[:,0] * wout/32
    wh0[:,1] = wh0[:,1] * hout/32

    wh_res = KMeans(n_clusters = args.n_clusters, random_state=0).fit(wh0)

    result = wh_res.cluster_centers_*32
    result = result[np.argsort(result[:,0])]
    str_anchors = ''
    for i,v in enumerate(result):
        if i == (args.n_clusters - 1):
            str_anchors += ' ' + str(int(v[0] + 0.5)) + ',' + str(int(v[1] + 0.5))
        else:
            str_anchors += ' ' + str(int(v[0] + 0.5)) + ',' + str(int(v[1] + 0.5)) + ','
    print(str_anchors)
    
    write_dict('input_size',args.input_size)
    write_dict('n_clusters',args.n_clusters)
    write_dict('anchors', str_anchors)


if __name__ == '__main__':
    print('compute anchors...')
    parser = argparse.ArgumentParser('compute anchors')
    parser.add_argument('--data_path', type=str, default='./darknet/dataset/img/', help='data path')
    parser.add_argument("--n_clusters", type=int, default=6, help="number of clusters")
    parser.add_argument("--input_size", type=int, default=608, help="The size of input")
    arguments = parser.parse_args()
    main(args=arguments)