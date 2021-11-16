#!/usr/bin/env python
# coding: utf-8

# Some core funtions in VFedPAC and VFedAKPCA 
# @Time   : 2021/11/06
# @Author : Feng Yu, Juyong Jiang
# @Email  : fengyu.sophia@gmail.com, csjuyongjiang@gmail.com

import numpy as np
import copy
import sys
from scipy.spatial.distance import pdist, squareform

# Federated algorithm
# 'wo' decides whether close weight scales federated learning
def federated(args, d_list, a_arr, v_arr):
    if args.de_ct:
        dec_pattern = input('Please choose the type of Decentralized Architecture from [full, ring, part]!\n>>>')
        if dec_pattern == 'full':
            print("Warning: You are using Fully Decentralized Architecture! ")
            final = dec_federated_full(d_list, a_arr, v_arr, args.wo_ws)

            return final
        elif dec_pattern == 'ring':
            print("Warning: You are using Ring Decentralized Architecture!")
            final = dec_federated_ring(d_list, a_arr, v_arr, args.wo_ws)
            
            return final
        elif dec_pattern == 'part':
            print("Warning: You are using Part Decentralized Architecture!")
            final = dec_federated_part(d_list, a_arr, v_arr, args.wo_ws)
            
            return final
        else:
            print("ERROR: This Decentralized Architecture doesn't be implemented! ")
            sys.exit(1)
    else:
        if args.wo_ws:
            print("Warning: You don't use Weight Scaling Method, istead of Average!")
            a_arr = [1.0/len(a_arr) for i in range(len(a_arr))]
        # the weight of each client based on eigenvalue
        v_w = (a_arr / np.sum(a_arr)).flatten() # be a vector
        print("The weight scales: ", v_w)
        # input('check')
        # re-weight the importance of each client's eigenvector (v_w * v_arr)
        B = [np.dot(k, v) for k, v in zip(v_w, v_arr)]
        # federated vector u as shared projection feature vector
        u = np.sum(B, axis=0)
        
        final_list = []
        for d in d_list:
            final_list.append(d.T.dot(u.T) if not args.kpca else d.T.dot(d).dot(u.T).T)
        final = np.hstack(final_list)
        # print("Federated PCA Final shape: ", final.shape)
        
        return final

def dec_federated_full(d_list, a_arr, v_arr, wo=False):
    def node_cent(node_i, a_arr, v_arr, wo=False):
        copy_earr = copy.deepcopy(a_arr)
        copy_varr = copy.deepcopy(v_arr)
        copy_earr.pop(node_i)
        copy_varr.pop(node_i)
        if wo:
            copy_earr = [1.0/len(copy_earr) for i in range(len(copy_earr))]
        # the weight of each client based on eigenvalue
        v_w = (copy_earr / (np.sum(copy_earr)+1e-9)).flatten()
        print("The full-dec weight scales: ", v_w)
        # input('check')
        # re-weight the importance of each client's eigenvector (v_w * v_arr)
        B = [np.dot(k, v) for k, v in zip(v_w, copy_varr)]
        # federated vector u as shared projection feature vector
        node_u = np.sum(B, axis=0)

        return node_u

    final_list = []
    for i in range(len(d_list)):
        node_u = node_cent(i, a_arr, v_arr)
        final_list.append(d_list[i].T.dot(node_u.T))
    final = np.hstack(final_list)
    # print("Federated PCA Final shape: ", final.shape)

    return final

def dec_federated_ring(d_list, a_arr, v_arr, wo=False):
    def node_cent(node_i, a_arr, v_arr, wo=False):
        copy_earr = copy.deepcopy(a_arr)
        copy_varr = copy.deepcopy(v_arr)
        start_idx, end_idx = (node_i-1) % len(copy_earr), (node_i+1) % len(copy_earr)

        ring_earr = [copy_earr[start_idx], copy_earr[end_idx]]
        ring_varr = [copy_varr[start_idx], copy_varr[end_idx]]

        if wo:
            ring_earr = [1.0/len(ring_earr) for i in range(len(ring_earr))]
        # the weight of each client based on eigenvalue
        v_w = (ring_earr / (np.sum(ring_earr)+1e-9)).flatten()
        print("The ring-dec weight scales: ", v_w)
        # input('check')
        # re-weight the importance of each client's eigenvector (v_w * v_arr)
        B = [np.dot(k, v) for k, v in zip(v_w, ring_varr)]
        # federated vector u as shared projection feature vector
        node_u = np.sum(B, axis=0)

        return node_u

    final_list = []
    for i in range(len(d_list)):
        node_u = node_cent(i, a_arr, v_arr)
        final_list.append(d_list[i].T.dot(node_u.T))
    final = np.hstack(final_list)
    # print("Federated PCA Final shape: ", final.shape)

    return final


# The number of clients should be 8 + 3m in all datasets
def dec_federated_part(d_list, a_arr, v_arr, wo=False):
    def node_cent(node_i, a_arr, v_arr, wo=False):
        copy_earr = copy.deepcopy(a_arr)
        copy_varr = copy.deepcopy(v_arr)

        if node_i < 3:
            neighb_earr  = [copy_earr[3]]
            neighb_varr = [copy_varr[3]]
        elif node_i == 3:
            neighb_earr = [copy_earr[i] for i in range(node_i-3, node_i, 1)]
            neighb_earr.append(copy_earr[int(2*node_i+1)])
            neighb_varr = [copy_varr[i] for i in range(node_i-3, node_i, 1)]
            neighb_varr.append(copy_varr[int(2*node_i+1)])
        elif 3 < node_i < 7:
            neighb_earr  = [copy_earr[7]]
            neighb_varr = [copy_varr[7]]
        else:
            neighb_earr = [copy_earr[i] for i in range(node_i-3, node_i, 1)]
            neighb_earr.append(copy_earr[int(node_i//2)])
            neighb_varr = [copy_varr[i] for i in range(node_i-3, node_i, 1)]
            neighb_varr.append(copy_varr[int(node_i//2)])
        
        if wo:
            neighb_earr = [1.0/len(neighb_earr) for i in range(len(neighb_earr))]
        # the weight of each client based on eigenvalue
        v_w = (neighb_earr / (np.sum(neighb_earr)+1e-9)).flatten()
        print("The part-dec weight scales: ", v_w)
        # input('check')
        # re-weight the importance of each client's eigenvector (v_w * v_arr)

        B = [np.dot(k, v) for k, v in zip(v_w, neighb_varr)]
        # federated vector u as shared projection feature vector
        node_u = np.sum(B, axis=0)

        return node_u

    final_list = []
    for i in range(len(d_list)):
        node_u = node_cent(i, a_arr, v_arr)
        final_list.append(d_list[i].T.dot(node_u.T))
    final = np.hstack(final_list)
    # print("Federated PCA Final shape: ", final.shape)

    return final


def isolated(d_list, max_eigv_list):
    isolated_list = []
    for d, e in zip(d_list, max_eigv_list):
        isolated_list.append(d.T.dot(e.T))
    isolated_vec = np.hstack(isolated_list)

    return isolated_vec

# Advanced Kernel PCA method
def AKPCA(data, n_dims, kernel_name):
    #:param data: (n_samples, n_features)
    #:param n_dims: target n_dims
    #:param kernel: kernel functions
    #:return: (n_samples, n_dims)
    kernel = get_kernel(kernel_name)
    
    K = kernel(data) # X*X.T=[n_samples, n_samples]
    N = K.shape[0] # the number of samples
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    eig_values, eig_vector = np.linalg.eig(K)
    idx = eig_values.argsort()[::-1] # get the index of sorted list by decreasing

    eigval = eig_values[idx][:n_dims] # get the maximum eigenvalues
    eigval = eigval**(1/2)

    eigvector = eig_vector[:, idx][:, :n_dims] # [n, 1]
    vi = eigvector / (eigval.reshape(-1, n_dims)+1e-9)

    data_n = np.dot(data.T, vi)

    return eigval, eigvector.reshape(-1), data_n

# Kernel PCA method
def KPCA(data, n_dims, kernel_name):
    #:param data: (n_samples, n_features)
    #:param n_dims: target n_dims
    #:param kernel: kernel functions
    #:return: (n_samples, n_dims)
    kernel = get_kernel(kernel_name)

    K = kernel(data.T)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    eig_values, eig_vector = np.linalg.eig(K)
    idx = eig_values.argsort()[::-1]

    eigval = eig_values[idx][:n_dims]
    eigval = eigval**(1/2)

    eigvector = eig_vector[:, idx][:, :n_dims]
    vi = eigvector / (eigval.reshape(-1, n_dims)+1e-9)

    # data_n = np.dot(K, vi) # optional

    return eigval, vi, vi

# PCA method
def PCA(data, k=1):
    mean_vec = np.mean(data, axis=0)
    scaled_x = data - mean_vec
    data_adjust = np.abs(scaled_x.T)
    cov_mat = np.cov(scaled_x)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    index = np.argsort(-eig_vals)
    selectVal = np.matrix(eig_vals.T[index[:k]])
    a = np.array(selectVal)
    selectVec = np.matrix(eig_vecs.T[index[:k]])
    v = np.array(selectVec)
    V = np.abs(v)
    finalData = scaled_x.T.dot(v.T)

    return finalData

r'''
    Local power iteration processing with random or warm start v
    to get eigenvalue and eigenvector in different communication period
'''
# Local power iteration algorithm -> eigenvector
def power_iteration(X, num_simu, v=None, first=None):
    A = np.cov(X)
    # start with a random vector then iteratively compute 
    b_k = np.random.rand(A.shape[1]) if not first else v
    for _ in range(num_simu):
        # eigenvector
        a_bk = np.dot(A, b_k)
        b_k = a_bk / (np.linalg.norm(a_bk)+1e-9)

        # eigenvalue
        e_k = np.dot(A, b_k.T).dot(b_k) / np.dot(b_k.T, b_k)

    return e_k, b_k

r'''
    Kernel function used in Kernel PCA or Advanced Kernel PCA (AKPCA)
'''
def get_kernel(kernel):
    if kernel == 'sigmoid':
        print("You are using kernel function: ", kernel)
        kernel = sigmoid

        return kernel
    elif kernel == 'rbf':
        print("You are using kernel function: ", kernel)
        kernel = rbf
        
        return kernel
    elif kernel == 'poly':
        print("You are using kernel function: ", kernel)
        kernel = poly
        return kernel
    else:
        print("ERROR: Kernel function not exist!")
        sys.exit(1)
    
def sigmoid(x, coef=0.25):
    x = np.dot(x, x.T)
    
    return np.tanh(coef*x+1)

def rbf(x, gamma=15):
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)

    return np.exp(-gamma * mat_sq_dists)

def poly(x, degree=3, c=1):
    x = np.dot(x, x.T) + c
    x = np.power(x, degree)
    return x