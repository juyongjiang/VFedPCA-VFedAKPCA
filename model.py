#!/usr/bin/env python
# coding: utf-8

# Some core funtions in VFedPAC and VFedAKPCA 
# @Time   : 2021/11/06
# @Author : Feng Yu, Juyong Jiang
# @Email  : 2063616201@qq.com, csjuyongjiang@gmail.com

import math
import time
import numpy as np

# Federated algorithm
def federated(a_arr, v_arr):
    # the weight of each client based on eigenvalue
    v_w = a_arr / np.sum(a_arr)
    # re-weight the importance of each client's eigenvector (v_w * v_arr)
    B = [np.dot(k, v) for k, v in zip(v_w, v_arr)]
    # federated vector u as shared projection feature vector
    u = np.sum(B, axis=0)

    return u

def pac_federated_ws(centers_list, k=10):
    res_centers_list = []
    for centers in centers_list:
        res_list, a_list, V_list, final_list = [], [], [], []
        for d in centers:
            a, V, final = pca(d, k)
            a_list.append(a)
            V_list.append(V)
            final_list.append(final)
        for i in range(k):
            res = federated_ws([a[:, i] for a in a_list], [v[:, i] for v in V_list])
            res_list.append(res)
        res_centers_list.append(res_list) #[[res1, res2,..., res_k], ...]


def federated_ws(a_arr, v_arr):
    # the weight of each client based on eigenvalue
    v_w = a_arr / np.sum(a_arr)
    # re-weight the importance of each client's eigenvector (v_w * v_arr)
    B = [v*k for k, v in zip(v_w, v_arr)]
    # federated vector u as shared projection feature vector
    u = np.sum(B, axis=0)

    return u

def pca(data, k=10):
    a, V, finaldata = [], [], []

    scaled_x = data - np.mean(data, axis=0)
    data_adjust = np.abs(scaled_x.T)

    cov_mat = np.cov(scaled_x)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    index = np.argsort(-eig_vals) # decrease

    selectVal=np.matrix(eig_vals.T[index[:k]]) # top_k
    a = np.array(selectVal)

    selectVec = np.matrix(eig_vecs.T[index[:k]])
    V = np.abs(np.array(selectVec))

    finaldata = data_adjust.dot(V.T)
    return a, V, finaldata

# Local power iteration processing or with warm start v
def max_eigen(A, iter_num=100, v=None, first=None):
    mat_cov = np.cov(A)
    max_eigs = mat_maxeigs(mat_cov)
    max_eigv = power_iteration(mat_cov, iter_num, v, first)
    
    return max_eigs, max_eigv

# Calculate the largest eigenvalue of the matrix -> eigenvalue
def mat_maxeigs(mat):
    [eigs, vecs] = np.linalg.eig(mat)
    abs_eigs = list(abs(eigs))
    max_id = abs_eigs.index(max(abs_eigs))
    # print('Largest eigenvalue of matrix: ', eigs[max_id])

    return eigs[max_id]

# Local power iteration algorithm -> eigenvector
def power_iteration(A, num_simu, v=None, first=None):
    # start with a random vector then iteratively compute 
    b_k = np.random.rand(A.shape[1]) if not first else v
    for _ in range(num_simu):
        # calculate the data matrix multiply eigenvector (A*b_k)
        a_bk = np.dot(A, b_k)
        # re normalize the vector
        b_k = a_bk / np.linalg.norm(a_bk)

    return b_k

# Calculate the distance error between two eigenvector (square)
def squared(a, b):
    d = 0;
    for i in range(len(a)):
        d += math.pow((a[i] - b[i]), 2) / (a[i] + b[i]);

    return d

def get_dis_time(max_eigv_list, d_list, p_list, centers_list, iter_num_list):
    err_list, time_list = [], []
    for p_index in range(len(p_list)):
        centers_p = centers_list[p_index]
        num_list = len(centers_list[p_index][0])
        err_p_list = []
        time_p_list = []
        for i in range(len(centers_p)):
            start_time = time.time()
            
            # get the eigenvalue and eigenvector for each d
            ep, vp = max_eigen(d_list[i], iter_num=iter_num_list[p_index])

            ep_arr, vp_arr = [], []
            for j in range(num_list):
                # get the eigenvalue and eigenvector of each period
                e_item, v_item = max_eigen(centers_p[i][j], iter_num=iter_num_list[0])
                ep_arr.append(e_item)
                vp_arr.append(v_item)
            # the federated vector
            rdp = federated(ep_arr, vp_arr)

            #compare the unsplited vector and federated vector error
            err_p = squared(vp, rdp)
            err_p_list.append(err_p)

            #calculate the time consuming
            time_p_list.append(time.time() - start_time)

        # compare the whole dataset vector error and time consuming
        start_time = time.time()
        err_arr_list = [squared(vp_arr[i], max_eigv_list[p_index]) for i in range(num_list)]
        avg_err_arr = (np.sum(err_arr_list)) / num_list
        time_arr = time.time() - start_time
        
        # error and time array
        err_p_list.insert(0, avg_err_arr)
        time_p_list.insert(0, time_arr)

        err_list.append(err_p_list)
        time_list.append(time_p_list)

    return err_list, time_list

def get_dis_ws(d_list, p_list, centers_list, inter_num=5):
    def get_res(centers, last_res=None, first=None):
        e_list, v_list = [], []
        for center_d in centers:
            e, v = max_eigen(center_d, iter_num=100, v=last_res, first=first)
            e_list.append(e)
            v_list.append(v)
        res_u = federated(e_list, v_list)

        return res_u

    err_list = []
    for p_index in range(len(p_list)):
        err_p_list = []
        for centers, d in zip(centers_list, d_list):
            ge, gv = max_eigen(d)
            last_res, first = None, False
            for i in range(inter_num):
                res_i = get_res(centers, last_res, first)
                last_res = res_i
                first = True
            err_p_center = squared(gv, last_res)
            err_p_list.append(err_p_center)
        
        err_list.append(err_p_list)

    return err_list