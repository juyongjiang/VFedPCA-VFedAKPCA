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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset/College.csv')

    parser.add_argument('--p_list', type=int, default=[3, 5, 10], help='the number of involved clients')
    parser.add_argument('--iter_list', type=list, default=[100, 100, 100], help='the number of local power iterations')
    parser.add_argument('--period_num', type=int, default=5)

    # synthetic dataset
    parser.add_argument('--u_list', type=int, default=[0, 4, 8])
    parser.add_argument('--sigma_list', type=list, default=[0.2, 0.6, 0.8])
    parser.add_argument('--shape', type=list, default=[20000, 4000])

    parser.add_argument('--warm_start', '-w', action='store_true', default=False, help='decide use warm start Method')
    parser.add_argument('--synthetic', '-s' action='store_true', default=False, help='decide use synthetic data')
    parser.add_argument('--flag', type=str, default='p')
    parser.add_argument('--show', action='store_true', default=False, help='decide display image in terminal')
    args = parser.parse_args()
    
    # To show the results of the given option to screen.
    for _, value in parser.parse_args()._get_kwargs():
        if value is not None:
            print(value)
    input('check')

    assert len(args.p_list) == len(args.iter_list), print("ERROR: len(args.p_list) != len(args.iter_list)!")

    '''
        Real-world Dataset
    '''
    data_name = args.data_path.split('/')[-1].split('.')[0]
    data_value = pd.read_csv(args.data_path, header=None, sep=',') # [row_num, fea_num]
    print("The name of dataset: ", data_name)
    print("The shape of dataset: ", data_value.shape) 
    
    '''
        Synthetic Dataset
    '''
    if args.synthetic:
        print("Warning: you are using synthetic dataset!")
        data_value, data_name = get_guas_data(args.u_list, args.sigma_list, args.shape, data_name='')


    # Sampling the dataset with dynamic_incremental_data_num
    dynamic_incremental_data_num = int(data_value.shape[0] / args.period_num)
    da_value = utils.arr_split(data_value, dynamic_incremental_data_num)
    print("The number of sampling: ", len(da_value)) # [dynamic_incremental_data_num, fea_num]
    # Each d is sampler with (dynamic_incremental_data_num * period, fea_num)
    d_list = utils.get_concat_data(da_value, args.period_num) 
    print("The shape of each d: ", [d.shape for d in d_list])
    
    centers_list = utils.get_centers_data(d_list, args.p_list) # [[d0, d1, d2, d3, d4], [d0, d1, d2, d3, d4],...]
    # centers_list_pd = utils.get_centers_data_pd([data_value], args.p_list)
    '''
        local power iteration process
    '''
    if not args.warm_start:
        max_eigv_list = []
        for iter_num in args.iter_list:
            max_eigs, max_eigv = model.max_eigen(data_value, iter_num) # the largest eigenvalue and eigenvector of the cov
            max_eigv_list.append(max_eigv)

        err_list, time_list = model.get_dis_time(max_eigv_list, d_list, args.p_list, centers_list, args.iter_list)
        print('Error convergence: ', err_list)
        print('Time consuming: ', time_list)
        utils.draw_fig(data_name, args.period_num, args.p_list, err_list, time_list, args.iter_list, flag, args.show)

    if args.warm_start:
        print("Warning: You are using warm start method!")
        err_ws_list = model.get_dis_ws(d_list, args.p_list, centers_list[0])
        utils.draw_fig_single(data_name, args.period_num, args.p_list, err_ws_list, flag, args.show)