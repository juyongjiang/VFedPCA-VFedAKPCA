#!/usr/bin/env python
# coding: utf-8

# Some core funtions in VFedPAC and VFedAKPCA 
# @Time   : 2021/11/06
# @Author : Feng Yu, Juyong Jiang
# @Email  : fengyu.sophia@gmail.com, csjuyongjiang@gmail.com

import math
import time
import numpy as np

def get_dis_time(args, data_value, clients_data_list):
    '''
        Step 1: Get Global Maximum Eigenvector By Using the Non-split Dataset
    '''
    global_eigs_list, global_eigv_list = [], []
    for i, iter_num in enumerate(args.iter_list):
        data_value = data_value[:, :int(data_value.shape[-1]/args.p_list[i])*args.p_list[i]] # ignore mismatch
        global_eigs, global_eigv = local_power_iteration(args, data_value, iter_num=iter_num, com_time=0, warm_start=None) 
        global_eigv_list.append(global_eigv)
        global_eigs_list.append(global_eigs)

    '''
        Step 2: Multi-shot Federated Learning with VFedAKPCA
    '''
    fed_dis_list, fed_time_list = [], []
    for p_idx in range(len(args.p_list)):
        dis_p_list, time_p_list = [], [] # [[dis_1, dis_2, dis_3, ..., dis_c], ...], c=communication period
        d_list = clients_data_list[p_idx] # [d1, d2, ...d_p], d_p=[n, fea_num]
        p_num = args.p_list[p_idx]
        ep_list, vp_list = [], []

        # start multi-shot federated learning
        start_time = time.time()
        
        if args.warm_start:
            print("Warning: you are using Local Power Iteration with Warm Start!")
        fed_u = None
        for cp in range(args.period_num+1):
            # get the eigenvalue and eigenvector for each client d
            for i in range(p_num):
                ep, vp = local_power_iteration(args, d_list[i], iter_num=args.iter_list[p_idx], \
                                        com_time=cp, warm_start=fed_u)
                ep_list.append(ep)
                vp_list.append(vp)

            if cp == 0:
                print("Warning: isolate period!")
                isolate_u = isolate(ep_list, vp_list)
                dis_p = squared_dis(global_eigv_list[p_idx], isolate_u)
                dis_p_list.append(dis_p)
                time_p_list.append(0)
                continue

            # federated vector
            fed_u = federated(ep_list, vp_list, args.weight_scale) # weight scale method

            # the global vector (from non-split dataset) and federated vector distance
            dis_p = squared_dis(global_eigv_list[p_idx], fed_u)
            dis_p_list.append(dis_p)
            
            # update local dataset for each clients
            rs_fed_u = np.expand_dims(fed_u, axis=-1)
            for i in range(p_num):
                # way 1
                mid_up = d_list[i].T.dot(rs_fed_u)
                up_item = mid_up.dot(mid_up.T) 
                up_item_norm = up_item / (np.linalg.norm(up_item)+1e-9)
                d_list[i] = d_list[i].dot(up_item_norm)

                # way 2
                # up_item = rs_fed_u.dot(rs_fed_u.T)
                # up_item_norm = up_item / np.linalg.norm(up_item)
                # d_list[i] = up_item_norm.dot(d_list[i])
            
            time_p_list.append(time.time() - start_time)

        fed_dis_list.append(dis_p_list)
        fed_time_list.append(time_p_list)

    return fed_dis_list, fed_time_list

# Calculate the distance error between 
# vfed and global eigenvectors (square)
def squared_dis(a, b, r=2.0):
    d = sum(((a[i] - b[i]) ** r) for i in range(len(a))) ** (1.0/r)

    # d = 0
    # for i in range(len(a)):
    #     d += math.pow((a[i] - b[i]), 2) / (a[i] + b[i])

    return d

# Federated algorithm
def federated(ep_list, vp_list, weight_scale):
    # the weight of each client based on eigenvalue
    v_w = ep_list / np.sum(ep_list)
    if weight_scale:
        print("Warning: you are using weight scaling method!")
        eta = np.mean(v_w) #
        en_num = len(ep_list) // 2 # the number of enhance clients
        idx = np.argsort(-v_w) # descending sort
        print("Before: ", v_w) 
        for i in idx[:en_num]:
            v_w[i] = (1+eta)*v_w[i]
        for j in idx[en_num:]:
            v_w[j] = (1-eta)*v_w[j]
        print("After: ", v_w)

    # re-weight the importance of each client's eigenvector (v_w * v_arr)
    B = [np.dot(k, v) for k, v in zip(v_w, vp_list)]
    # federated vector u as shared projection feature vector
    u = np.sum(B, axis=0)

    return u

# isolate algorithm
def isolate(ep_list, vp_list):
    ep_avg = [1.0 for i in range(len(ep_list))]
    # the weight of each client based on eigenvalue
    v_w = ep_avg / np.sum(ep_avg)
    B = [np.dot(k, v) for k, v in zip(v_w, vp_list)]
    # federated vector u as shared projection feature vector
    u = np.sum(B, axis=0)

    return u

# Local power iteration processing or with warm start v 
def local_power_iteration(args, X, iter_num, com_time, warm_start):
    A = np.cov(X)
    # start with a random vector or warm start v
    judge_use = com_time not in [0, 1] and args.warm_start
    b_k = warm_start if judge_use else np.random.rand(A.shape[0])

    for _ in range(iter_num):
        # eigenvector
        a_bk = np.dot(A, b_k) 
        b_k = a_bk / (np.linalg.norm(a_bk)+1e-9) #
        
        # eigenvalue
        e_k = np.dot(A, b_k.T).dot(b_k) / np.dot(b_k.T, b_k)

    return e_k, b_k