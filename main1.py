#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:35:52 2023

@author: weiliu
"""
import numpy as np
import logging
import argparse
import time
from scipy.sparse.linalg import eigsh, eigs
import argparse
from read_data import load_2um, load_15um, load_30um, load_60um, Generate_Materials_Data
from model1 import model_nnet, model_logis
from cov_estimation1 import SARM_method, SARM_method1, direct_method
import os

def Brutal_select_window_size(scores, scores_hat, wind_size_choice, n_wid, n_hei, add_mat_ls, A, diag):
    def Add_Nug_Mat(sym_mat, nugget=1e-8, rcond=1e-5):
        """Add conditional number onto a symmetric matrix to ensure the condition number."""
        eigs, _ = np.linalg.eig(sym_mat)
        min_eig = np.real(eigs[np.argmin(eigs)])
        if min_eig > 0:
            eig_large, _ = eigsh(
                sym_mat, 1, which='LM')  # find k eigenvalues and eigenvectors
            iden_factor = eig_large[0]*rcond
            # logger.info("The largest eigen_value is {}.\n".format(eig_large), extra=d)
            logger.info("The identity matrix need to add: a nugget %s and the factor to ensure conditional number is %s.", nugget, iden_factor, extra=d)
            nugget = max(nugget, iden_factor)
        else:
            nugget = -min_eig + nugget
        return sym_mat + nugget*np.identity(sym_mat.shape[0])

    def Inv_Mat_Rcond(sym_mat, nugget=1e-8, rcond=1e-5):
        """Invert a matrix to ensure a condition number."""
        sym_mat = Add_Nug_Mat(sym_mat, nugget, rcond)
        return np.linalg.inv(sym_mat)

    def Inv_Cov(score_vecs, nugget=1e-8, rcond=1e-5, wei=None):
        """Invert covariance matrix of the score"""
        if wei is None:
            wei = np.ones((score_vecs.shape[0], 1))
        score_mu = np.sum(score_vecs, axis=0, dtype=np.float32)/np.sum(wei)
        score_centered = score_vecs - score_mu*wei
        S = np.dot(np.transpose(score_centered),
                   score_centered/wei) / np.sum(wei)
        return Inv_Mat_Rcond(S)

    def HotellingT2(x, mu, Sinv, dtype=np.float32):
        """ Calculate Hotelling T2 statitic for a specific pixel. """
        tmp = x - mu
        tmp = np.reshape(tmp, (1, Sinv.shape[0]))
        return np.dot(np.dot(tmp, Sinv), np.transpose(tmp))

    mu = np.mean(scores, axis=0)
    var_score = np.matmul(scores_hat.T, scores_hat,
                          dtype=np.float32)/scores.shape[0]
    window_t2_1 = [None] * len(wind_size_choice)
    window_t2_2 = [None] * len(wind_size_choice)
    window_t2_3 = [None] * len(wind_size_choice)
    scores_arr = scores.reshape((n_wid, n_hei, scores.shape[1]))

    round = 0
    for i in range(len(wind_size_choice)):
        w = wind_size_choice[i]
        n_w = 2*w + 1
        N_w = (2*w+1)**2
        start = time.time()
        start_inner = time.time()
        ij_t2_1 = [None] * ((n_wid-2*w)*(n_wid-2*w))
        ij_t2_2 = [None] * ((n_wid-2*w)*(n_wid-2*w))
        ij_t2_3 = [None] * ((n_wid-2*w)*(n_wid-2*w))
        count_inner = 0
        w_scores = np.zeros(
            ((n_wid-2*w)*(n_wid-2*w), scores.shape[1]), dtype=np.float32)
        # used to update the window moves right
        store_wscore_right = np.zeros((scores.shape[1],), dtype=np.float32)
        # used to update the window moves down
        store_wscore_down = np.zeros((scores.shape[1],), dtype=np.float32)
        for ri in range(w, n_hei-w):
            for ci in range(w, n_wid-w):
                count_inner += 1
                if ri == w and ci == w:
                    window_score = np.average(
                        scores_arr[(ri-w):(ri+w+1), (ci-w):(ci+w+1)], axis=(0, 1))
                    w_scores[(ri-w)*(n_wid-2*w)+ci-w] = window_score
                    store_wscore_right = window_score
                    store_wscore_down = window_score
                elif ci == w:
                    window_score = store_wscore_down + np.average(scores_arr[ri+w, (ci-w):(ci+w+1)], axis=0)/(
                        2*w+1) - np.average(scores_arr[ri-w-1, (ci-w):(ci+w+1)], axis=0)/(2*w+1)
                    w_scores[(ri-w)*(n_wid-2*w)+ci-w] = window_score
                    store_wscore_right = window_score
                    store_wscore_down = window_score
                else:
                    window_score = store_wscore_right + np.average(scores_arr[(ri-w):(ri+w+1), ci+w], axis=0)/(
                        2*w+1) - np.average(scores_arr[(ri-w):(ri+w+1), ci-w-1], axis=0)/(2*w+1)
                    w_scores[(ri-w)*(n_wid-2*w)+ci-w] = window_score
                    store_wscore_right = window_score
        # logger.info("The inner area calculation takes {}min\n".format((time.time()-start_inner)//60), extra=d)
        '''
        start_outer = time.time()
        count_outer = 0
        # starts the outer area

        if w == 0:
            cov_mat = var_score
            Sinv = Inv_Mat_Rcond(cov_mat)
        else:
            add_mat = add_mat_ls[i]
            cov_mat = (N_w*var_score + add_mat)/N_w**2
            Sinv = Inv_Mat_Rcond(cov_mat)
            logger.info("add_mat is \n {}".format(add_mat), extra=d)
            logger.info("before inversion is \n {}".format(cov_mat), extra=d)
            logger.info("after inversion is \n {}".format(Sinv), extra=d)
            eigs, _ = np.linalg.eig(Sinv)
            logger.info("max eigen value after inversion {}".format(
                np.real(eigs[np.argmax(eigs)])), extra=d)
        for i in range((n_wid-2*w)*(n_wid-2*w)):
            ij_t2[i] = HotellingT2(w_scores[i], mu, Sinv, dtype=np.float32)
        window_t2[round] = ij_t2
        round += 1
        logger.info("T^2 statistic for windows with size: {}, takes total time: {}min, mean is {}".format(
            w, (time.time()-start)//60, np.mean(np.array(ij_t2))), extra=d)
        # logger.info("Inner area has {} windows, outer area has {} windows, in total: {}".format(count_inner, count_outer, count_inner+count_outer), extra=d)
        '''
        if A == 'One':
            for i in range((n_wid-2*w)*(n_wid-2*w)):
                ij_t2_1[i] = np.inner(w_scores[i] - mu, w_scores[i] - mu)
                ij_t2_2[i] = ij_t2_1[i] * n_w
                ij_t2_3[i] = ij_t2_1[i] * N_w
        elif A == 'NoCorrect':
            if diag == False:
                cov_mat1 = var_score
                cov_mat2 = var_score*n_w / n_w**2
                cov_mat3 = var_score*N_w / N_w**2
            else:
                cov_mat1 = np.diag(np.diag(var_score))
                cov_mat2 = np.diag(np.diag(var_score))*n_w / n_w**2
                cov_mat3 = np.diag(np.diag(var_score))*N_w / N_w**2
             
            Sinv1 = Inv_Mat_Rcond(cov_mat1)
            Sinv2 = Inv_Mat_Rcond(cov_mat2)
            Sinv3 = Inv_Mat_Rcond(cov_mat3)
            for i in range((n_wid-2*w)*(n_wid-2*w)):
                ij_t2_1[i] = HotellingT2(w_scores[i], mu, Sinv1, dtype=np.float32)
                ij_t2_2[i] = HotellingT2(w_scores[i], mu, Sinv2, dtype=np.float32)
                ij_t2_3[i] = HotellingT2(w_scores[i], mu, Sinv3, dtype=np.float32)
        elif A == 'Correct':
            if diag == False:
                if w == 0:
                    add_mat = 0
                else:
                    add_mat = add_mat_ls[i]
                cov_mat1 = var_score + add_mat_ls[i]
                cov_mat2 = (var_score*n_w + add_mat_ls[i]) / n_w**2
                cov_mat3 = (var_score*N_w + add_mat_ls[i]) / N_w**2
            else:
                if w == 0:
                    add_mat = 0
                else:
                    add_mat = add_mat_ls[i]
                cov_mat1 = np.diag(np.diag(var_score + add_mat_ls[i]))
                cov_mat2 = np.diag(np.diag(var_score*n_w + add_mat_ls[i])) / n_w**2
                cov_mat3 = np.diag(np.diag(var_score*N_w + add_mat_ls[i])) / N_w**2
             
            Sinv1 = Inv_Mat_Rcond(cov_mat1)
            Sinv2 = Inv_Mat_Rcond(cov_mat2)
            Sinv3 = Inv_Mat_Rcond(cov_mat3)
            for i in range((n_wid-2*w)*(n_wid-2*w)):
                ij_t2_1[i] = HotellingT2(w_scores[i], mu, Sinv1, dtype=np.float32)
                ij_t2_2[i] = HotellingT2(w_scores[i], mu, Sinv2, dtype=np.float32)
                ij_t2_3[i] = HotellingT2(w_scores[i], mu, Sinv3, dtype=np.float32)
        window_t2_1[round] = ij_t2_1
        window_t2_2[round] = ij_t2_2 
        window_t2_3[round] = ij_t2_3
        round += 1   
    return window_t2_1, window_t2_2, window_t2_3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RVE size determination algorithm')
    # '2um', '15um', '30um', '60um'
    parser.add_argument('--data', type=str, default='2um',
                        help='which dataset to use')
    # 'nnet', 'logis'
    parser.add_argument('--model', type=str, default='nnet',
                        help='which model to use')
    # 'SARM', 'direct', 'CLT', 'All'
    parser.add_argument('--cov_method', type=str, default='All',
                        help='which covariance estimation method to use')
    parser.add_argument('--filename', type=str,
                        default='std', help='output filename')
    args = parser.parse_args()

    # read arguments
    data_name = args.data
    model_type = args.model
    cov_method = args.cov_method

    FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s(%(funcName)s)[%(lineno)d]: %(message)s'
    logging.basicConfig(filename="{}.log".format(
        args.filename), format=FORMAT, filemode='w')
    d = {'clientip': '192.168.0.1', 'user': 'lw'}
    logger = logging.getLogger('main')
    logging.getLogger('main').setLevel(logging.INFO)

    logger.info("dataset: {}, model: {}, cov_method: {}".format(
        data_name, model_type, cov_method), extra=d)

    # read data from micrograph file
    if data_name == '2um':
        img_arr = load_2um()
        wid_ls = [0, 20, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    elif data_name == '15um':
        img_arr = load_15um()
        wid_ls = [0, 50, 75, 100, 150, 200, 250,
                  300, 350, 400, 450, 500, 550, 600]
    elif data_name == '30um':
        img_arr = load_30um()
        wid_ls = [0, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 275, 300]
    elif data_name == '60um':
        img_arr = load_60um()
        wid_ls = [0, 30, 50, 75, 100, 125, 150, 175,
                  200, 225, 250, 300, 350, 400, 450, 500]
    
    # obtain training data X,y for model fitting
    # pre specify wind_hei=wind_wid=10, tune if necessary
    wind_hei, wind_wid = 10, 10
    data_X, data_y, n_hei, n_wid = Generate_Materials_Data(
        img_arr, wind_hei=wind_hei, wind_wid=wind_wid)
    minor_idx = [i for i in range(data_X.shape[0]) if data_y[i]==1]
    major_idx = [i for i in range(data_X.shape[0]) if i not in minor_idx]
    minor_X, minor_y = data_X[minor_idx], data_y[minor_idx]
    data_Xmajor, data_ymajor = data_X[major_idx], data_y[major_idx]
    minor_idx_upsample = np.random.randint(0, len(minor_idx), 600000)
    data_Xminor, data_yminor = minor_X[minor_idx_upsample], minor_y[minor_idx_upsample]
    data_X = np.concatenate((data_Xmajor, data_Xminor), axis=0)
    data_y = np.concatenate((data_ymajor, data_yminor))
    logger.info("The micrograph size (in pixel) is {}x{}.\n".format(
        img_arr.shape[0], img_arr.shape[1]), extra=d)
    logger.info("The data size is {}, {}.\n".format(
        data_X.shape, data_y.shape), extra=d)

    filename = data_name + "_" + model_type + "_" + "scores.npy"
    logger.info("Looking for filename {}".format(filename), extra=d)
    logger.info("{}".format(os.path.isfile('./'+filename)), extra=d)
    if not os.path.isfile('./'+filename):
     # compute score vectors based on different models
        logger.info("Model type is {}".format(model_type), extra=d)
        if model_type == 'nnet':
            scores, scores_hat = model_nnet(data_X, data_y)
        elif model_type == 'logis':
            scores, scores_hat = model_logis(data_X, data_y)
        var_score = np.matmul(scores_hat.T, scores_hat,
                              dtype=np.float32)/scores.shape[0]
        max_eiv, _ = eigsh(var_score, 1, which='LM')
        logger.info("scores matrix as maximum eigenvalue {}, norm {}".format(
            max_eiv, np.linalg.norm(var_score)), extra=d)
        logger.info("scores cov matrix {}".format(var_score), extra=d)
        with open(filename, 'wb') as f:
            np.save(f, scores)
    else:
        with open(filename, "rb") as f:
            scores = np.load(filename)

    scores_hat = scores - np.mean(scores, axis=0)
    var_scores = np.matmul(scores_hat.T, scores_hat, dtype=np.float32)/scores.shape[0]
    max_eiv, _ = eigsh(var_scores, 1, which='LM')
     
    # obtain list of covariance matrices and com
    logger.info("Inner product", extra=d)
    t2_ls_1, t2_ls_2, t2_ls_3 = Brutal_select_window_size(scores, scores_hat, wid_ls, n_wid, n_hei, add_mat_ls = None, A = 'One', diag=False)
    for t2_ls in [t2_ls_1, t2_ls_2, t2_ls_3]:
        t2_ls_msd = np.array([np.mean(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Average of T_2 by CLT is {}'.format(t2_ls_msd), extra=d)
        t2_ls_sd = np.array([np.std(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Std of T_2 by CLT is {}'.format(t2_ls_sd), extra=d)
        t2_ls_qua = np.array([np.quantile(t2_ls[i], [0.95]) for i in range(len(t2_ls))])
        logger.info("Upper quantile is {}".format(t2_ls_qua), extra=d)

    logger.info("No correct, no diag", extra=d)
    t2_ls_1, t2_ls_2, t2_ls_3 = Brutal_select_window_size(scores, scores_hat, wid_ls, n_wid, n_hei, add_mat_ls = None, A = 'NoCorrect', diag=False)
    for t2_ls in [t2_ls_1, t2_ls_2, t2_ls_3]:
        t2_ls_msd = np.array([np.mean(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Average of T_2 by CLT is {}'.format(t2_ls_msd), extra=d)
        t2_ls_sd = np.array([np.std(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Std of T_2 by CLT is {}'.format(t2_ls_sd), extra=d)
        t2_ls_qua = np.array([np.quantile(t2_ls[i], [0.95]) for i in range(len(t2_ls))])
        logger.info("Upper quantile is {}".format(t2_ls_qua), extra=d)

    logger.info("No correct, diag", extra=d)
    t2_ls_1, t2_ls_2, t2_ls_3 = Brutal_select_window_size(scores, scores_hat, wid_ls, n_wid, n_hei, add_mat_ls = None, A = 'NoCorrect', diag=True)
    for t2_ls in [t2_ls_1, t2_ls_2, t2_ls_3]:
        t2_ls_msd = np.array([np.mean(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Average of T_2 by CLT is {}'.format(t2_ls_msd), extra=d)
        t2_ls_sd = np.array([np.std(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Std of T_2 by CLT is {}'.format(t2_ls_sd), extra=d)
        t2_ls_qua = np.array([np.quantile(t2_ls[i], [0.95]) for i in range(len(t2_ls))])
        logger.info("Upper quantile is {}".format(t2_ls_qua), extra=d)
    '''
    start_time = time.time()
    cov_ls, cut_point, eig_ls = direct_method(
        scores, scores_hat, n_wid, n_hei, wid_ls, max_eiv, tole=1e-3)
    logger.info('coef computation takes time :{}min, cut point is h=k={}'.format(
       (time.time() - start_time)//60, cut_point), extra=d)
    logger.info('eig_ls is {}'.format(eig_ls), extra=d)
 
    logger.info("Correct by direct, no diag", extra=d)
    t2_ls_1, t2_ls_2, t2_ls_3 = Brutal_select_window_size(scores, scores_hat, wid_ls, n_wid, n_hei, add_mat_ls = cov_ls, A = 'Correct', diag=False)
    for t2_ls in [t2_ls_1, t2_ls_2, t2_ls_3]:
        t2_ls_msd = np.array([np.mean(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Average of T_2 by CLT is {}'.format(t2_ls_msd), extra=d)
        t2_ls_sd = np.array([np.std(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Std of T_2 by CLT is {}'.format(t2_ls_sd), extra=d)
        t2_ls_qua = np.array([np.quantile(t2_ls[i], [0.95]) for i in range(len(t2_ls))])
        logger.info("Upper quantile is {}".format(t2_ls_qua), extra=d)

    logger.info("Correct by direct, diag", extra=d)
    t2_ls_1, t2_ls_2, t2_ls_3 = Brutal_select_window_size(scores, scores_hat, wid_ls, n_wid, n_hei, add_mat_ls = cov_ls, A = 'Correct', diag=True)
    for t2_ls in [t2_ls_1, t2_ls_2, t2_ls_3]:
        t2_ls_msd = np.array([np.mean(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Average of T_2 by CLT is {}'.format(t2_ls_msd), extra=d)
        t2_ls_sd = np.array([np.std(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Std of T_2 by CLT is {}'.format(t2_ls_sd), extra=d)
        t2_ls_qua = np.array([np.quantile(t2_ls[i], [0.95]) for i in range(len(t2_ls))])
        logger.info("Upper quantile is {}".format(t2_ls_qua), extra=d)
    
    cov_ls, cut_point, eig_ls = SARM_method(
        scores, scores_hat, n_wid, n_hei, wid_ls, max_eiv, tole=1e-3)
    logger.info('coef computation takes time :{}min, cut point is h=k={}'.format(
       (time.time() - start_time)//60, cut_point), extra=d)
    logger.info('eig_ls is {}'.format(eig_ls), extra=d)
 
    logger.info("Correct by direct, no diag", extra=d)
    t2_ls_1, t2_ls_2, t2_ls_3 = Brutal_select_window_size(scores, scores_hat, wid_ls, n_wid, n_hei, add_mat_ls = cov_ls, A = 'Correct', diag=False)
    for t2_ls in [t2_ls_1, t2_ls_2, t2_ls_3]:
        t2_ls_msd = np.array([np.mean(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Average of T_2 by CLT is {}'.format(t2_ls_msd), extra=d)
        t2_ls_sd = np.array([np.std(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Std of T_2 by CLT is {}'.format(t2_ls_sd), extra=d)
        t2_ls_qua = np.array([np.quantile(t2_ls[i], [0.95]) for i in range(len(t2_ls))])
        logger.info("Upper quantile is {}".format(t2_ls_qua), extra=d)

    logger.info("Correct by direct, diag", extra=d)
    t2_ls_1, t2_ls_2, t2_ls_3 = Brutal_select_window_size(scores, scores_hat, wid_ls, n_wid, n_hei, add_mat_ls = cov_ls, A = 'Correct', diag=True)
    for t2_ls in [t2_ls_1, t2_ls_2, t2_ls_3]:
        t2_ls_msd = np.array([np.mean(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Average of T_2 by CLT is {}'.format(t2_ls_msd), extra=d)
        t2_ls_sd = np.array([np.std(t2_ls[i]) for i in range(len(t2_ls))])
        logger.info('Std of T_2 by CLT is {}'.format(t2_ls_sd), extra=d)
        t2_ls_qua = np.array([np.quantile(t2_ls[i], [0.95]) for i in range(len(t2_ls))])
        logger.info("Upper quantile is {}".format(t2_ls_qua), extra=d)
    ''' 
