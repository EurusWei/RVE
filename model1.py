#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:51:45 2023

@author: weiliu
"""

import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import logging
import os
import pickle

logger = logging.getLogger('main.model')
d = {'clientip': '192.168.0.1', 'user': 'lw'}
# tune the model if necessary


def model_nnet(data_X, data_y):
    '''
    if not os.path.isfile('./nnet.sav'):
        logger.info("Model does not exist, fitting right now\n", extra=d)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(10,), random_state=1)
        clf.fit(data_X, data_y)
        filename = 'nnet.sav'
        pickle.dump(clf, open(filename, 'wb'))
    else:
        logger.info("Reload model now\n")
        clf = pickle.load(open('nnet.sav', 'rb'))
    '''
    logger.info("Enter the model function", extra=d)
    n = data_X.shape[0]
    test_size = int(n * 0.33)
    logger.info("test size {}".format(test_size), extra=d)
    all_idx = np.arange(n)
    np.random.shuffle(all_idx)
    logger.info("{}".format(all_idx[:5]), extra=d)
    test_idx = all_idx[:test_size]
    train_idx = all_idx[test_size:]
    logger.info("train size {}".format(len(train_idx)), extra=d)
    X_train, y_train = data_X[train_idx, :], data_y[train_idx]
    X_test, y_test = data_X[test_idx, :], data_y[test_idx]
    logger.info("start fitting the model", extra=d)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(10,), max_iter=1000, random_state=1)
    clf.fit(X_train, y_train)
    # calculate score vectors against parameters of the last layer
    logger.info("Accuracy of this model is {}".format(
        clf.score(X_test, y_test)), extra=d)
    pred_y = clf.predict_proba(data_X)
    data_X = np.hstack((data_X, np.ones((data_X.shape[0], 1))))
    coef_hidden = np.vstack((clf.coefs_[0], clf.intercepts_[0]))
    activation_X = np.inner(data_X, coef_hidden.T)
    activation_X = np.hstack(
        (activation_X, np.ones((activation_X.shape[0], 1))))
    #default using sign function
    sign = np.sign(activation_X)
    activation_X *= sign
    grads = np.zeros((pred_y.shape[0], activation_X.shape[-1]))
    for i in range(pred_y.shape[0]):
        grads[i] = (pred_y[i][1]-data_y[i]) * activation_X[i]
    scores = -grads
    scores = np.array(scores, dtype=np.float32)
    scores_hat = scores - np.mean(scores, axis=0)
    return scores, scores_hat

def model_nnet1(data_X, data_y):
    '''
    if not os.path.isfile('./nnet.sav'):
        logger.info("Model does not exist, fitting right now\n", extra=d)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(10,), random_state=1)
        clf.fit(data_X, data_y)
        filename = 'nnet.sav'
        pickle.dump(clf, open(filename, 'wb'))
    else:
        logger.info("Reload model now\n")
        clf = pickle.load(open('nnet.sav', 'rb'))
    '''
    logger.info("Enter the model function", extra=d)
    n = data_X.shape[0]
    test_size = int(n * 0.33)
    logger.info("test size {}".format(test_size), extra=d)
    all_idx = np.arange(n)
    np.random.shuffle(all_idx)
    logger.info("{}".format(all_idx[:5]), extra=d)
    test_idx = all_idx[:test_size]
    train_idx = all_idx[test_size:]
    logger.info("train size {}".format(len(train_idx)), extra=d)
    X_train, y_train = data_X[train_idx, :], data_y[train_idx]
    X_test, y_test = data_X[test_idx, :], data_y[test_idx]
    logger.info("start fitting the model", extra=d)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(10,), max_iter=1000, random_state=1)
    clf.fit(X_train, y_train)
    # calculate score vectors against parameters of the last layer
    logger.info("Accuracy of this model is {}".format(
        clf.score(X_test, y_test)), extra=d)
    pred_y = clf.predict_proba(data_X)
    data_X = np.hstack((data_X, np.ones((data_X.shape[0], 1))))
    input_d = data_X.shape[1]
    coeff_1 = np.vstack((clf.coefs_[0], clf.intercepts_[0]))
    coeff_2 = np.vstack((clf.coefs_[1], clf.intercepts_[1]))
    
    activation_X = np.inner(data_X, coeff_1.T)
    activation_X = np.hstack(
        (activation_X, np.ones((activation_X.shape[0], 1))))
    hidden_d = activation_X.shape[1]
    grads = np.zeros((pred_y.shape[0], (hidden_d-1)*input_d + hidden_d))
    design_mat = np.zeros((hidden_d-1, data_X.shape[1]*(hidden_d-1)))
    for i in range(hidden_d - 1):
        design_mat[i, i*data_X.shape[1]: (i+1)*data_X.shape[1]] = np.ones(input_d)
    for i in range(pred_y.shape[0]):
        grads[i, (hidden_d-1)*input_d:] = (pred_y[i][1]-data_y[i]) * activation_X[i]
        a = (pred_y[i][1]-data_y[i]) * coeff_2
        a = a[:-1]
        b = np.diag(np.sign(activation_X[i][:-1]))
        c = np.tile(data_X[i], (hidden_d-1)) * design_mat
        grads[i, :(hidden_d-1)*input_d] = np.inner(c.T, np.inner(a.T, b)).squeeze()
    scores = -grads
    scores = np.array(scores, dtype=np.float32)
    scores_hat = scores - np.mean(scores, axis=0)
    return scores, scores_hat

def model_logis(data_X, data_y):
    n = data_X.shape[0]
    test_size = int(n * 0.33)
    logger.info("test size {}".format(test_size), extra=d)
    all_idx = np.arange(n)
    np.random.shuffle(all_idx)
    logger.info("{}".format(all_idx[:5]), extra=d)
    test_idx = all_idx[:test_size]
    train_idx = all_idx[test_size:]
    logger.info("train size {}".format(len(train_idx)), extra=d)
    X_train, y_train = data_X[train_idx, :], data_y[train_idx]
    X_test, y_test = data_X[test_idx, :], data_y[test_idx]
    logger.info("start fitting the model", extra=d)
    clf = LogisticRegression(solver='lbfgs', random_state=1, max_iter=10000)
    clf.fit(X_train, y_train)
    logger.info("Accuracy of this model is {}".format(
        clf.score(X_test, y_test)), extra=d) 
    # calculate score vectors against parameters of the last layer
    pred_y = clf.predict_proba(data_X)
    data_X = np.hstack((data_X, np.ones((data_X.shape[0], 1))))
    grads = np.zeros((pred_y.shape[0], data_X.shape[-1]))
    for i in range(pred_y.shape[0]):
        grads[i] = (pred_y[i][1]-data_y[i]) * data_X[i]
    scores = -grads
    scores = np.array(scores, dtype=np.float32)
    scores_hat = scores - np.mean(scores, axis=0)
    return scores, scores_hat
