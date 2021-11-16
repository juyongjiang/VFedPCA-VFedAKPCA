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
import copy
import argparse
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset/GlaucomaM.csv')

    parser.add_argument('--p_list', type=list, default=[3, 5, 10], help='the number of involved clients')
    parser.add_argument('--iter_list', type=list, default=[10, 10, 10], help='the number of local power iterations')
    parser.add_argument('--period_num', type=int, default=10)
    parser.add_argument('--sample_num', type=int, default=None)

    # synthetic dataset
    parser.add_argument('--pattern', type=str, default='mixture') # mixture or single
    parser.add_argument('--shape', type=list, default=[20000, 4000]) # M features x N samples
    parser.add_argument('--img_path', type=str, default='./figs')

    parser.add_argument('--warm_start', '-w', action='store_true', default=False, help='decide use warm start Method')
    parser.add_argument('--weight_scale', '-r', action='store_true', default=False, help='decide use warm start Method')
    parser.add_argument('--synthetic', '-s', action='store_true', default=False, help='decide use synthetic data')
    parser.add_argument('--label', type=str, default='p', help='[p, l, ws]')
    parser.add_argument('--show', action='store_true', default=False, help='decide display image in terminal')
    args = parser.parse_args()
    
    '''
        Training Config
    '''
    print("*******Training Config*******")
    for key, value in parser.parse_args()._get_kwargs():
        if value is not None:
            print(key, '=', value)

    assert len(args.p_list) == len(args.iter_list), print("ERROR: len(args.p_list) != len(args.iter_list)!")
    print('****************************')
    
    '''
        Real-world Dataset
    '''
    if not args.synthetic:
        data_name = args.data_path.split('/')[-1].split('.')[0]
        data_value = pd.read_csv(args.data_path, header=None, sep=',')[:args.sample_num] if args.sample_num \
                                                        else pd.read_csv(args.data_path, header=None, sep=',') # [row_num, fea_num]
        print("The name of dataset: ", data_name)
        print("The shape of dataset: ", data_value.shape) 
    
    '''
        Synthetic Dataset
    '''
    if args.synthetic:
        print("Warning: you are using synthetic dataset!")
        data_value, data_name = utils.get_guas_data(args.pattern, args.shape, data_name=args.shape)

    #
    data_value = np.array(data_value)

    print("The number of client: ", len(args.p_list)) # [data_num, fea_num]
    clients_data_list = utils.get_client_data(args, data_value) # [[d1, d2, ...d_p], ...]
    for i in range(len(args.p_list)):
        client_data = clients_data_list[i]
        print("# clients = ",args.p_list[i])
        for data in client_data:
            print(data.shape)

    '''
        Local Power Iteration Process with Warm Start
    '''
    fed_dis_list, fed_time_list = model.get_dis_time(args, copy.deepcopy(data_value), copy.deepcopy(clients_data_list))

    print('Dis(V, U): ', fed_dis_list)
    print('Time     : ', fed_time_list)
    
    utils.draw_fig(args, data_name, fed_dis_list, fed_time_list, fig_path=args.img_path)
