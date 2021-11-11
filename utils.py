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

# Split the dataset 
def arr_split(arr, size):
    s = []
    for i in range(0, len(arr)+1, size):
        c = arr[i: i+size]
        s.append(c)

    return s

def get_concat_data(da1, sampler_num):
    cat_list = []
    d0 = np.array(da1[0])
    cat_list.append(d0)

    for i in range(sampler_num - 1):
        d = np.array(da1[i+1])
        cat = np.vstack((cat_list[-1], d)) # concat with dim=0
        cat_list.append(cat)

    return cat_list


def get_centers_data(d_list, p_list):
    fea_num = d_list[0].shape[-1]
    centers_list = []
    for i in range(len(p_list)): 
        tmp_list = []
        fea_split = round(fea_num / p_list[i])
        num_list = int(fea_num / fea_split)
        for j in range(len(d_list)):
            centers_d = []
            for k in range(num_list):
                centers_d.append(d_list[j][:, k*fea_split:(k+1)*fea_split])
            tmp_list.append(centers_d)
        centers_list.append(tmp_list)

    return centers_list
    # centers0 = [d0[:, 0:5], d0[:, 5:10], d0[:, 10:15]]
    # centers1 = [d1[:, 0:5], d1[:, 5:10], d1[:, 10:15]]
    # centers2 = [d2[:, 0:5], d2[:, 5:10], d2[:, 10:15]]
    # centers3 = [d3[:, 0:5], d3[:, 5:10], d3[:, 10:15]]
    # centers4 = [d4[:, 0:5], d4[:, 5:10], d4[:, 10:15]]
    # centers_list[[centers0, centers1, centers2, centers3, centers4],...]

def get_centers_data_pd(d_list, p_list):
    fea_num = d_list[0].shape[-1]
    centers_list = []
    for i in range(len(p_list)): 
        tmp_list = []
        fea_split = round(fea_num / p_list[i])
        num_list = int(fea_num / fea_split)
        for j in range(len(d_list)):
            centers_d = []
            for k in range(num_list):
                centers_d.append(d_list[j].iloc[:, k*fea_split:(k+1)*fea_split])
            tmp_list.append(centers_d)
        centers_list.append(tmp_list)

    return centers_list

def draw_fig(data_name, sampler_num, p_list, err_list, time_list, iter_list, flag='p', show=True, fig_path='./figs'):
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    font2 = {'family' : 'Times New Roman',
    'weight' : 'bold',
    'size'   : 16,
    }

    x1 = np.arange(sampler_num+1)
    x2 = np.arange(sampler_num+1)
    color = ['g', 'r', 'b']

    # Error
    fig_1, ax1 = plt.subplots()
    xy1 = plt.gca()
    xy1.spines['bottom'].set_linewidth(2)
    xy1.spines['left'].set_linewidth(2)
    xy1.spines['right'].set_linewidth(2)
    xy1.spines['top'].set_linewidth(2)
    # ax2 = ax1.twinx()   
    ax1.set_title(data_name, font2) 
    ax1.set_xlabel('Communication Period', font2)    
    ax1.set_ylabel('Dis(V,U)', font2)   

    for i in range(len(p_list)):        
        ax1.plot(x1, err_list[i], color[i], ls='-', label=flag+'='+str(p_list[i]), linewidth=2, marker='s')
    # fig_1.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.legend(loc='upper left')
    # plt.savefig(os.path.join(fig_path, data_name + '_' + flag + '.eps'), format='eps', dpi=1000)
    plt.savefig(os.path.join(fig_path, data_name + '_error_' + flag + '.png'), format='png')
    plt.show()

    # time
    fig_2, ax2 = plt.subplots()
    xy2 = plt.gca()
    xy2.spines['bottom'].set_linewidth(2)
    xy2.spines['left'].set_linewidth(2)
    xy2.spines['right'].set_linewidth(2)
    xy2.spines['top'].set_linewidth(2) 
    ax2.set_title(data_name, font2) 
    ax2.set_xlabel('Communication Period', font2)     
    ax2.set_ylabel('Time /s', font2)   

    for i in range(len(p_list)):
        ax2.plot(x2, time_list[i], color[i], ls='--', label=flag+'='+str(p_list[i]), linewidth=2, marker='s')
    # fig_2.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.legend(loc='upper left')
    # plt.savefig(os.path.join(fig_path, data_name + '_' + flag + '.eps'), format='eps', dpi=1000)
    plt.savefig(os.path.join(fig_path, data_name + '_time_' + flag + '.png'), format='png')
    plt.show()

def draw_fig_single(data_name, sampler_num, p_list, err_list, flag, show=True, fig_path='./figs'):
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    font2 = {'family' : 'Times New Roman',
    'weight' : 'bold',
    'size'   : 16,
    }

    x3 = np.arange(0, sampler_num, 1)
    color = ['g', 'r', 'b']

    # Error
    fig, ax1 = plt.subplots()
    xy = plt.gca()
    xy.spines['bottom'].set_linewidth(2)
    xy.spines['left'].set_linewidth(2)
    xy.spines['right'].set_linewidth(2)
    xy.spines['top'].set_linewidth(2)
    plt.xticks(x3)
    ax1.set_title(data_name, font2) 
    ax1.set_xlabel('Communication Period', font2)    
    ax1.set_ylabel('Dis(V,U)', font2)   
    
    for i in range(len(p_list)): 
        err_list[i].reverse()
        ax1.plot(x3, err_list[i], color[2], ls='-', linewidth=2)

    # fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    # plt.savefig(os.path.join(fig_path, data_name + '_' + flag + '.eps'), format='eps', dpi=1000)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(fig_path, data_name + '_' + flag + '.png'), format='png', dpi=1000)
    plt.show()

def get_guas_data(u_list, sigma_list, shape, data_name=''):
    n, m = shape
    each_num = n * m / len(u_list)
    
    y_list = []
    for u, sigma in zip(u_list, sigma_list):
        print("u = ", u, "sigma = ", sigma)
        x = np.linspace(u-5*sigma, u+5*sigma, int(each_num))
        y = np.exp(-(x - u) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma)
        y_list.append(y)

    y_arr = np.hstack(y_list)
    assert y_arr.shape[-1] == n*m, print('ERROR: The number of synthetic data is not correct!') 
    index = np.random.permutation(y_arr.size)
    y_random_arr = y_arr[index]
    y_matrix = y_random_arr.reshape((n, m))
    print("The shape of synthetic matrix is: ", y_matrix.shape)
    
    return y_matrix, data_name

    