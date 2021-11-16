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
from skimage.transform import resize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/Image/Face/Face_1')

    parser.add_argument('--client_num', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--re_size', type=int, default=100)
    parser.add_argument('--kernel', type=str, default='sigmoid')

    parser.add_argument('--n_clusters', type=int, default=10)

    parser.add_argument('--kpca', action='store_true', default=False, help='decide use Kernel PCA Method')
    parser.add_argument('--wo_ws', action='store_true', default=False, help='decide not use weight scaling method in federated communication')
    parser.add_argument('--de_ct', action='store_true', default=False, help='decide use decentralized architecture in federated communication')
    parser.add_argument('--show', action='store_true', default=False, help='decide display image in terminal')
    args = parser.parse_args()
    
    # print information about dataset
    data_name = args.data_path.split('/')[-1]
    print("The name of dataset: ", data_name)
    imgpath_list = sorted([os.path.join(args.data_path, file) for file in os.listdir(args.data_path)])
    print("The number of images: ", len(imgpath_list))
    
    # read all images as numpy array
    resarr_list = []
    for path in imgpath_list:
        # to cal kpca, scale img to small size
        img = np.asarray(imageio.imread(path))
        img = resize(img, (args.re_size, args.re_size))

        img_size = img.shape
        img = img.flatten() # (w, h) -> (w*h,)
        resarr_list.append(img)
    print("The size of each image: ", img_size)

    # obtain the whole dataset eigenvalue and eigenvector
    f = np.asarray(resarr_list) # [img_num, w*h]
    max_eigs_f, max_eigv_f = model.power_iteration(f, args.iterations)

    # vertically partition dataset 
    d_list = utils.get_split_data(f, args.client_num) # each clients with [img_num, w*h/client_num]
    print('-----------------------------------')
    print("The shape of each d: ", [d.shape for d in d_list])
    # get each clients' eigenvalue and eigenvector, respectively
    max_eigs_list, max_eigv_list = utils.get_eig_data(d_list, args.iterations) 

    r'''###################################
                        PCA
        ###################################
    '''
    if not args.kpca:
        # unsplitted pca
        us_pca = f.T.dot(max_eigv_f.T) 
    
        # splitted VFedPCA 
        vfed_pca = model.federated(args, d_list, max_eigs_list, max_eigv_list)
        
        # isolated PCA
        is_pca = model.isolated(d_list, max_eigv_list) 
        
        # print shape to check
        print('-----------------------------------')
        print('The shape of us_pca: ', us_pca.shape)
        print('The shape of vfed_pca: ', vfed_pca.shape)
        print('The shape of is_pca: ', is_pca.shape)

        # draw each pca-sub figure
        utils.draw_subfig(us_pca, 'us_pca_' + data_name, args.re_size, args.show)
        utils.draw_subfig(vfed_pca, 'vfed_pca_' + data_name, args.re_size, args.show)
        utils.draw_subfig(is_pca, 'is_pca_' + data_name, args.re_size, args.show)

        # draw each cluster figure
        utils.draw_cluster(args.n_clusters, us_pca, 'us_pca_cluster_' + data_name, args.re_size, args.show)
        utils.draw_cluster(args.n_clusters, vfed_pca, 'vfed_pca_cluster_' + data_name, args.re_size, args.show)
        utils.draw_cluster(args.n_clusters, is_pca, 'is_pca_cluster_' + data_name, args.re_size, args.show)
    
    r'''###################################
                    VFedAKPCA 
        ###################################
    '''
    if not args.kpca:
        print("Warning: You are using A_KPCA method!")
        # unsplitted VFedAKPCA
        al, av, us_vfedak_pca = model.AKPCA(f, n_dims=1, kernel_name=args.kernel) # f=[img_num, w*h]
        # splitted VFedAKPCA 
        x_max_eigs_list, x_max_eigv_list, x_max_f_list = utils.get_x_pca_data(d_list, model.AKPCA, kernel_name=args.kernel, n_dims=1) 
    
    else:
        print("Warning: You are using KPCA method!")
        # unsplitted VFedKPCA
        al, av, us_vfedak_pca = model.KPCA(f, n_dims=1, kernel_name=args.kernel) # f=[img_num, w*h]
        # splitted VFedKPCA 
        x_max_eigs_list, x_max_eigv_list, x_max_f_list = utils.get_x_pca_data(d_list, model.KPCA, kernel_name=args.kernel, n_dims=1) 
    
    vfedak_pca = model.federated(args, d_list, x_max_eigs_list, x_max_eigv_list) 
    # isolated VFedAKPCA / VFedKPCA
    is_vfedak_pca = np.vstack(x_max_f_list) 
    
    print('The shape of us_vfedak_pca: ', us_vfedak_pca.shape)
    print('The shape of vfedak_pca: ', vfedak_pca.shape)
    print('The shape of is_vfedak_pca: ', is_vfedak_pca.shape)

    # draw each akpca/kpca-sub-figure
    utils.draw_subfig(us_vfedak_pca, 'us_vfedak_pca_'  + data_name, args.re_size, args.show)
    utils.draw_subfig(vfedak_pca, 'vfedak_pca_' + data_name, args.re_size, args.show)
    utils.draw_subfig(is_vfedak_pca, 'is_vfedak_pca_' + data_name, args.re_size, args.show)