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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('main.model')
d = {'clientip': '192.168.0.1', 'user': 'lw'}
# tune the model if necessary

#nnet using model in sklearn
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
                        hidden_layer_sizes=(10,), max_iter=1000, random_state=2)
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
    #default using relu activation function
    sign = np.sign(activation_X)
    nonneg_sign = np.where(sign > 0, sign, 0)
    activation_X *= nonneg_sign
    grads = np.zeros((pred_y.shape[0], activation_X.shape[-1]))
    for i in range(pred_y.shape[0]):
        grads[i] = (pred_y[i][1]-data_y[i]) * activation_X[i]
    scores = -grads
    scores = np.array(scores, dtype=np.float32)
    scores_hat = scores - np.mean(scores, axis=0)
    return scores, scores_hat

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        # how to turn a list of labels to a tensor
        # https://stackoverflow.com/questions/44617871/how-to-convert-a-list-of-strings-into-a-tensor-in-pytorch
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Nnet(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        """
        num_layers: number of layers in the PE network
        input_dim: dimension of input features
        hidden_dim: dimension of hidden layers
        output_dim: number of classes for prediction
        final_dropout: dropout ration in the final layer
        """
        super().__init__()
        self.input_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(feature_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

#nnet using pytorch
def model_nnet1(data_X, data_y):
    learning_rate = 0.001
    epochs = 10
    batch_size = 32
    hidden_dim = 10
    model = Nnet(data_X.shape[1], hidden_dim = hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 1e-5)
    n = data_X.shape[0]
    loss_fn = nn.BCELoss()

    #train the model
    train_data = MyDataset(data_X, data_y)
    trainloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)     #training loop
    losses = []
    accur = []
    for i in range(epochs):
        model.train()
        count_accu = 0
        for batch_idx,(x_train,y_train) in enumerate(trainloader):
            optimizer.zero_grad()
            #calculate output
            output = model(x_train)
            #calculate loss
            loss = loss_fn(output,y_train.reshape(-1,1))
            count_accu += (output.reshape(-1).detach().numpy().round() == np.array(y_train)).sum()
            #backprop
            loss.backward()
            optimizer.step()

        if i%100 == 0:
            losses.append(loss)
            accur.append(count_accu/n)
            logger.info("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,count_accu/n), extra = d)

    #calculate scores against parameters in the last layer
    testloader = DataLoader(train_data, batch_size = 1, shuffle = False)
    scores = np.ones((data_X.shape[0], hidden_dim + 1))
    model.eval()
    for i, (test_X, test_y) in enumerate(testloader):
        optimizer.zero_grad()
        output = model(test_X)
        grad = torch.autograd.grad(loss_fn(output, test_y), model.parameters())
        scores[i][:-1], scores[i][-1] = grad[-2], grad[-1]
    scores_hat = np.mean(scores, axis=0)
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
