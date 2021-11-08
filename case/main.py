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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/Image/Face/Face_1')
    parser.add_argument('--client_num', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--re_size', type=int, default=100)
    parser.add_argument('--show', action='store_true', default=False, help='decide whether display image in terminal')
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

    # print information about dataset
    data_name = args.data_path.split('/')[-1]
    print("The name of dataset: ", data_name)
    imgpath_list = sorted([os.path.join(args.data_path, file) for file in os.listdir(args.data_path)])
    print("The number of images: ", len(imgpath_list))
    
    # read all images as numpy array
    resarr_list = []
    for path in imgpath_list:
        img = np.asarray(imageio.imread(path))
        img_size = img.shape
        img = img.flatten() # (w, h) -> (w*h,)
        resarr_list.append(img)
    print("The size of each image: ", img_size)
    f = np.asarray(resarr_list) # [img_num, w*h]

    # vertically partition dataset 
    d_list = utils.get_split_data(f, args.client_num) # each clients with [img_num, w*h/client_num]
    print('-----------------------------------')
    print("The shape of each d: ", [d.shape for d in d_list])
    max_eigs_list, max_eigv_list = utils.get_eig_data(d_list, args.iterations)

    # algorithm start 
    max_eigs_f, max_eigv_f = model.max_eigen(f)
    # unsplitted pca
    us_pca = f.T.dot(max_eigv_f.T) 
    
    # VFedPCA 
    vfed_pca = model.federated(d_list, max_eigs_list, max_eigv_list)

    # isolated PCA
    is_pca = utils.get_final_data(d_list, max_eigv_list) 

    # VFedAKpca
    x_max_eigs_list, x_max_eigv_list, x_max_f_list = utils.get_xpca_data(d_list, model.AKpca, utils.rbf) 
    vfedak_pca = model.federated(d_list, x_max_eigs_list, x_max_eigv_list) 

    # unsplitted VFedAKpca
    al, av, us_vfedak_pca = model.AKpca(f, n_dims=1, kernel=utils.rbf) 

    # isolated VFedAKpca
    is_vfedak_pca = np.vstack(x_max_f_list) 
    
    # print shape to check
    print('-----------------------------------')
    print('The shape of us_pca: ', us_pca.shape)
    print('The shape of vfed_pca: ', vfed_pca.shape)
    print('The shape of is_pca: ', is_pca.shape)
    print('The shape of vfedak_pca: ', vfedak_pca.shape)
    print('The shape of us_vfedak_pca: ', us_vfedak_pca.shape)
    print('The shape of is_vfedak_pca: ', is_vfedak_pca.shape)

    # draw each sub-figure
    utils.draw_subfig(us_pca, 'us_pca_' + data_name, args.re_size, args.show)
    utils.draw_subfig(vfed_pca, 'vfed_pca_' + data_name, args.re_size, args.show)
    utils.draw_subfig(is_pca, 'is_pca_' + data_name, args.re_size, args.show)

    utils.draw_subfig(vfedak_pca, 'vfedak_pca_' + data_name, args.re_size, args.show)
    utils.draw_subfig(us_vfedak_pca, 'us_vfedak_pca_' + data_name, args.re_size, args.show)
    utils.draw_subfig(is_vfedak_pca, 'is_vfedak_pca_' + data_name, args.re_size, args.show)