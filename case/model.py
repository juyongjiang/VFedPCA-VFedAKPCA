#!/usr/bin/env python
# coding: utf-8

# Some core funtions in VFedPAC and VFedAKPCA 
# @Time   : 2021/11/06
# @Author : Feng Yu, Juyong Jiang
# @Email  : fengyu.sophia@gmail.com, csjuyongjiang@gmail.com

import numpy as np
import copy
from scipy.spatial.distance import pdist, squareform

# Federated algorithm
# 'wo' decides whether close weight scales federated learning
def federated(d_list, a_arr, v_arr, wo=False, de=False):
    if de:
        print("Warning: You are using Fully Decentralized Architecture!")
        final = dec_federated(d_list, a_arr, v_arr, wo)

        return final
    else:
        if wo:
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
            final_list.append(d.T.dot(u.T))
        final = np.hstack(final_list)
        # print("Federated PCA Final shape: ", final.shape)
        
        return final

def dec_federated(d_list, a_arr, v_arr, wo=False):
    def node_cent(node_i, a_arr, v_arr, wo=False):
        copy_earr = copy.deepcopy(a_arr)
        copy_varr = copy.deepcopy(v_arr)
        copy_earr.pop(node_i)
        copy_varr.pop(node_i)
        if wo:
            copy_earr = [1.0/len(copy_earr) for i in range(len(copy_earr))]
        # the weight of each client based on eigenvalue
        v_w = copy_earr / np.sum(copy_earr).flatten()
        print("The weight scales: ", v_w)
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
    vi = eigvector / eigval.reshape(-1, n_dims)

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
    vi = eigvector / eigval.reshape(-1, n_dims)

    data_n = np.dot(K, vi) # optional

    return eigval, eigvector.reshape(-1), data_n

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
        b_k = a_bk / (np.linalg.norm(a_bk)+1e-9)

    return b_k

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