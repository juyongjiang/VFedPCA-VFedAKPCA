#!/usr/bin/env python
# coding: utf-8

# Some utils in VFedPAC and VFedAKPCA 
# @Time   : 2021/11/06
# @Author : Feng Yu, Juyong Jiang
# @Email  : fengyu.sophia@gmail.com, csjuyongjiang@gmail.com

import os
import numpy as np
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as mp
from matplotlib.ticker import MaxNLocator
import random

# Split the dataset for each client
def get_client_data(args, arr):
    # arr=[n, fea_num] -> each_client=[n, fea_split_num])
    fea_num = arr.shape[-1] 
    data_list = []
    for p in args.p_list:
        fea_split_num = int(fea_num / p)
        tmp_list = [] 
        for j in range(p):
            p_j_data = arr[:, j*fea_split_num:(j+1)*fea_split_num]
            tmp_list.append(p_j_data) # round [n, fea_split_num]
        data_list.append(tmp_list)

    return data_list # [[d1, d2, ...d_p], ...]

def get_guas_data(pattern, shape, data_name=''):
    data_name = '[' + str(data_name[0]) + ', ' + str(data_name[1]) + ']'
    m, n = shape # M features x N samples
    u_list = [random.uniform(0, m) for i in range(m)] if pattern=='mixture' else [0 for i in range(m)]
    sigma_list = [random.uniform(0, m) for i in range(m)] if pattern=='mixture' else [1 for i in range(m)]

    each_num = n
    
    y_list = []
    for u, sigma in zip(u_list, sigma_list):
        print("u = ", u, "sigma = ", sigma)
        x = np.array([random.uniform(u-3*sigma, u+3*sigma) for i in range(each_num)]) # randomly sample x points
        y = np.exp(-(x - u) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma)
        y = np.expand_dims(y, axis=-1)
        y_list.append(y)

    y_matrix = np.hstack(y_list)

    assert y_matrix.shape[0]==n and y_matrix.shape[1] == m, print('ERROR: The number of synthetic data is not correct!') 
    print("The shape of synthetic matrix is: ", y_matrix.shape)
    
    return y_matrix, data_name # N x M

def draw_fig(args, data_name, fed_dis_list, fed_time_list, fig_path='./figs'):
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_num = args.p_list if args.label == 'p' else args.iter_list
    
    font2 = {'family' : 'Times New Roman',
    'weight' : 'bold',
    'size'   : 16,
    }

    x1 = [x for x in range(args.period_num+1)]
    x2 = [x for x in range(args.period_num+1)]
    color = ['g', 'r', 'b', 'y', 'm', 'c', 'r', 'b', 'g']
    marker = ['s', 'o', '*', 'd', '+', 'x', 'd', '+', 'x']

    # Error
    fig_1, ax1 = plt.subplots()
    xy1 = plt.gca()
    # plt.xticks(x1)
    # xy1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # xy1.yaxis.set_major_locator(MaxNLocator(integer=True))
    xy1.spines['bottom'].set_linewidth(2)
    xy1.spines['left'].set_linewidth(2)
    xy1.spines['right'].set_linewidth(2)
    xy1.spines['top'].set_linewidth(2)
    # ax2 = ax1.twinx()   
    ax1.set_title(data_name, font2) 
    ax1.set_xlabel('Communication Period', font2)    
    ax1.set_ylabel('Dis(V,U)', font2)   


    for i in range(len(fed_dis_list)):        
        ax1.plot(x1, fed_dis_list[i], color[i], ls='-', label=args.label+'='+str(fig_num[i%len(args.p_list)]), linewidth=2, marker=marker[i])
    fig_1.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(fig_path, data_name + '_error_' + args.label + '.eps'), format='eps', dpi=1000)
    plt.savefig(os.path.join(fig_path, data_name + '_error_' + args.label + '.png'), format='png', dpi=1000)
    plt.show()

    # time
    fig_2, ax2 = plt.subplots()
    xy2 = plt.gca()
    # plt.xticks(x2)
    # xy2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # xy2.yaxis.set_major_locator(MaxNLocator(integer=True))
    xy2.spines['bottom'].set_linewidth(2)
    xy2.spines['left'].set_linewidth(2)
    xy2.spines['right'].set_linewidth(2)
    xy2.spines['top'].set_linewidth(2) 
    ax2.set_title(data_name, font2) 
    ax2.set_xlabel('Communication Period', font2)     
    ax2.set_ylabel('Time /s', font2)   

    for i in range(len(fed_time_list)):
        ax2.plot(x2, fed_time_list[i], color[i], ls='--', label=args.label+'='+str(fig_num[i%len(args.p_list)]), linewidth=2, marker=marker[i])
    fig_2.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(fig_path, data_name + '_time_' + args.label + '.eps'), format='eps', dpi=1000)
    plt.savefig(os.path.join(fig_path, data_name + '_time_' + args.label + '.png'), format='png', dpi=1000)
    plt.show()
