#!/usr/bin/env python
# coding: utf-8

# The main function in VFedPAC and VFedAKPCA 
# @Time   : 2021/11/06
# @Author : Feng Yu, Juyong Jiang
# @Email  : fengyu.sophia@gmail.com, csjuyongjiang@gmail.com

import math
import model
import time
import os
import utils
import argparse
import pandas as pd
import numpy as np

from sklearn import datasets, preprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset/College.csv')
    parser.add_argument('--batch_size', type=int, default=160)
    parser.add_argument('--show', action='store_true', default=False, help='decide whether show results image in terminal')
    args = parser.parse_args()

    # EXP-1 
    # flag ='clients'
    # p_list = [3, 5, 10] # the number of involved clients
    # iter_list = [100, 100, 100] # the number of local power iterations
    # sampler_num = 5

    # EXP-2
    # flag ='iterations'
    # p_list = [5, 5, 5] # the number of involved clients
    # iter_list = [5, 10, 20] # the number of local power iterations
    # sampler_num = 5
    
    # EXP-3
    # flag ='warmstart'
    # p_list = [5] # the number of involved clients
    # iter_list = [100] # the number of local power iterations
    # sampler_num = 5

    data_name = args.data_path.split('/')[-1].split('.')[0]
    print("The name of dataset: ", data_name)
    
    if data_name in ['Swarm', 'TCGA']:
        p_list = [10, 50, 100]
        iter_list = [100, 100, 100]
        sampler_num = 10

    data1 = pd.read_csv(args.data_path, header=None, sep=',')
    print("The shape of dataset: ", data1.shape) # [row_num, fea_num]
    
    # Sampling the dataset with batch_size
    da1 = utils.arr_split(data1, args.batch_size)
    print("The number of sampling: ", len(da1)) # [batch_size, fea_num]

    # Each d is sampler with (batch size * id, fea_num)
    d_list = utils.get_concat_data(da1, sampler_num) 
    print("The shape of each d: ", [d.shape for d in d_list])
    
    centers_list = utils.get_centers_data(d_list, p_list) # [[d0, d1, d2, d3, d4], [d0, d1, d2, d3, d4],...]
    # centers_list_pd = utils.get_centers_data_pd([data1], p_list)
    '''
        local power iteration process
    '''
    if flag != 'warmstart':
        max_eigv_list = []
        for iter_num in iter_list:
            max_eigs, max_eigv = model.max_eigen(data1, iter_num) # the largest eigenvalue and eigenvector of the cov
            max_eigv_list.append(max_eigv)

        err_list, time_list = model.get_dis_time(max_eigv_list, d_list, p_list, centers_list, iter_list)
        print('Error convergence: ', err_list)
        print('Time consuming: ', time_list)
        utils.draw_fig(data_name, sampler_num, p_list, err_list, time_list, iter_list, flag, args.show)

    if flag == 'warmstart':
        err_ws_list = model.get_dis_ws(d_list, p_list, centers_list[0])
        utils.draw_fig_single(data_name, sampler_num, p_list, err_ws_list, args.show)