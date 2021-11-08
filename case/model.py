#!/usr/bin/env python
# coding: utf-8

# Some core funtions in VFedPAC and VFedAKPCA 
# @Time   : 2021/11/06
# @Author : Feng Yu, Juyong Jiang
# @Email  : fengyu.sophia@gmail.com, csjuyongjiang@gmail.com

import numpy as np

# Federated algorithm
def federated(d_list, a_arr, v_arr):
    # the weight of each client based on eigenvalue
    v_w = a_arr / np.sum(a_arr)
    # re-weight the importance of each client's eigenvector (v_w * v_arr)
    B = [np.dot(k, v) for k, v in zip(v_w, v_arr)]
    # federated vector u as shared projection feature vector
    u = np.sum(B, axis=0)
    
    final_list = []
    for d in d_list:
        final_list.append(d.T.dot(u.T))
    final = np.vstack(final_list)
    print("Federated PCA Final shape: ", final.shape)

    return final

# Local power iteration processing or with warm start v to get eigenvalue and eigenvector
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

#Advanced Kernel PCA method
def AKpca(data, n_dims, kernel):
    #:param data: (n_samples, n_features)
    #:param n_dims: target n_dims
    #:param kernel: kernel functions
    #:return: (n_samples, n_dims)
    K = kernel(data)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eig_values, eig_vector = np.linalg.eig(K)
    idx = eig_values.argsort()[::-1]
    eigval = eig_values[idx][:n_dims]
    eigvector = eig_vector[:, idx][:, :n_dims]
    eigval = eigval**(1/2)
    vi = eigvector / eigval.reshape(-1, n_dims)
    data_n = np.dot(data.T, vi)
    #print(data_n)
    #data_f=data.T.dot(data_n)

    return eigval, eigvector, data_n