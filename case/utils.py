#!/usr/bin/env python
# coding: utf-8

# Some utils in VFedPAC and VFedAKPCA 
# @Time   : 2021/11/06
# @Author : Feng Yu, Juyong Jiang
# @Email  : fengyu.sophia@gmail.com, csjuyongjiang@gmail.com

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import max_eigen
from sklearn.cluster import KMeans

def get_split_data(img_arr, client_num):
    fea_split = int(img_arr.shape[-1] / client_num)
    d_list = []
    for i in range(client_num):
        d_i = img_arr[:, i*fea_split:(i+1)*fea_split]
        d_list.append(d_i)

    return d_list

def get_eig_data(d_list, iterations):
    max_eigs_list, max_eigv_list = [], []
    for d in d_list:
        max_eigs, max_eigv = max_eigen(d, iterations)
        max_eigs_list.append(max_eigs)
        max_eigv_list.append(max_eigv)

    return max_eigs_list, max_eigv_list

# 'x' means that you can choose differnt func by name, e.g. AKPCA, KPCA, PCA
def get_x_pca_data(d_list, func, kernel_name, n_dims=1):
    max_eigs_list, max_eigv_list, max_f_list = [], [], []
    for d in d_list:
        max_eigs, max_eigv, max_f = func(d, n_dims, kernel_name)
        max_eigs_list.append(max_eigs.T)
        max_eigv_list.append(max_eigv.T)
        max_f_list.append(max_f)

    return max_eigs_list, max_eigv_list, max_f_list

def draw_subfig(result, name, re_size, show=True, path='./figs'):
    result = np.array(result, dtype='float64').reshape(re_size, re_size)
    fig, ax1 = plt.subplots()
    plt.axis('off')
    plt.imshow(result, interpolation = "none", cmap = "gray")
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, name)
    # plt.savefig(path+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    plt.savefig(path+'.png', format='png', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    plt.show()


def draw_cluster(num_k, result, name, re_size, show=True, path='./figs'):
    fig = plt.figure()
    img = np.array(result, dtype='float').reshape(re_size, re_size)
    kmeans = KMeans(n_clusters=num_k, random_state=0).fit(img)
    pic2show = kmeans.cluster_centers_[kmeans.labels_]
    plt.axis('off')
    plt.imshow(pic2show, cmap='gray')
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, name)
    # plt.savefig(path+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    plt.savefig(path+'.png', format='png', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    plt.show()