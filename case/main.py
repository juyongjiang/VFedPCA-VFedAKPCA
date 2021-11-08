#!/usr/bin/env python
# coding: utf-8

# The main function in VFedPAC and VFedAKPCA 
# @Time   : 2021/11/06
# @Author : Feng Yu, Juyong Jiang
# @Email  : fengyu.sophia@gmail.com, csjuyongjiang@gmail.com

import math
import utils
import os
import imageio
import argparse
import numpy as np
import model

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from scipy.spatial.distance import pdist, squareform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/Image/DeepLesion')
    parser.add_argument('--client_num', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--re_size', type=int, default=512)
    args = parser.parse_args()
    
    # cuhk03
    # client_num = 8
    # iterations = 10
    # args.re_size = 128

    # lung
    # client_num = 8
    # iterations = 100
    # args.re_size = 512

    # gait
    # client_num = 16
    # iterations = 50
    # args.re_size = 256
    
    # face
    # client_num = 10
    # iterations = 100
    # args.re_size = 100

    imgpath_list = sorted([os.path.join(args.data_path, file) for file in os.listdir(args.data_path)])
    print("The number of images: ", len(imgpath_list))
    
    resarr_list = []
    for path in imgpath_list:
        img = np.asarray(imageio.imread(path)).flatten() # (w, h) -> (w*h,)
        resarr_list.append(img)
    
    f = np.asarray(resarr_list) # [img_num, w*h]
    d_list = utils.get_split_data(f, args.client_num) # d1, d2, d3, d4, d5, d6, d7, d8
    max_eigs_list, max_eigv_list = utils.get_eig_data(d_list, args.iterations)

    # algorithm start 
    max_eigs_f, max_eigv_f = model.max_eigen(f)
    train_f = f.T.dot(max_eigv_f.T)

    final = model.federated(d_list, max_eigs_list, max_eigv_list)

    sf = utils.get_final_data(d_list, max_eigv_list)

    x_max_eigs_list, x_max_eigv_list, x_max_f_list = utils.get_xpca_data(d_list, model.AKpca, utils.rbf)
    aFinal = model.federated(d_list, x_max_eigs_list, x_max_eigv_list)

    al, av, af = model.AKpca(f, n_dims=1, kernel=utils.rbf)

    afinal = np.vstack(x_max_f_list)
    
    # print shape to check
    print('The shape of train_f: ', train_f.shape)
    print('The shape of final: ', final.shape)
    print('The shape of sf: ', sf.shape)
    print('The shape of aFinal1: ', aFinal.shape)
    print('The shape of af: ', af.shape)
    print('The shape of afinal: ', afinal.shape)

    # draw each sub-figure
    utils.draw_subfig(train_f, 'cug1', args.re_size)
    utils.draw_subfig(final, 'cuf1', args.re_size)
    utils.draw_subfig(sf, 'cus1', args.re_size)

    utils.draw_subfig(aFinal, 'kcuf1', args.re_size)
    utils.draw_subfig(af, 'kcug1', args.re_size)
    utils.draw_subfig(afinal, 'kcus1', args.re_size)