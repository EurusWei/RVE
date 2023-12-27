#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:22:11 2023

@author: weiliu
"""

import numpy as np
import scipy.io as sio
import logging

def load_2um():
    #load 2um micrograph and return in array
    def load_image(path):
        im = sio.loadmat(path)
        return im
    
    im = load_image('../data/2um.mat')
    img_arr = im['newMS']
    return img_arr

def load_15um():
    #load 15um micrograph and return in array
    def load_image(path):
        im = sio.loadmat(path)
        return im
    
    im = load_image('../data/flakes15.mat')
    img_arr = im['ReconImg']
    return img_arr

def load_30um():
    #load 30um micrograph and return in array
    def load_image(path):
        im = sio.loadmat(path)
        return im
    
    im = load_image('../data/flakes30.mat')
    img_arr = im['ReconImg']
    return img_arr

def load_60um():
    #load 60um micrograph and return in array
    def load_image(path):
        im = sio.loadmat(path)
        return im
    
    im = load_image('../data/TrainData1#(1).mat')
    img_arr = im['TrainData'][0][0][2]
    return img_arr

#generate array data from the image
def Generate_Materials_Data(img_arr, wind_hei, wind_wid):
    """Generate materials micro-structure data from image array.
       The returned X, y are filled in row-by-row in the image pixel order."""
    img_hei, img_wid = img_arr.shape
    n_hei = (img_hei - 2 * wind_hei) 
    n_wid = (img_wid - 2 * wind_wid)
    n_sample = n_hei * n_wid
    row_idx = 0
    
    #initiate the array
    xy_dim = (2 * wind_hei+1) * (2 * wind_wid+1)
    Xy = np.zeros((n_sample, xy_dim))
    
    for ci in range(wind_wid, img_wid - wind_wid): #iterate start from the left of the inner frame
        #fix jth column in the first row
        #reshape.() flattens the array and piece together row by row
        xy_wind_ls = np.reshape(img_arr[:(2*wind_hei+1), (ci-wind_wid):(ci+wind_wid+1)], (-1,))
        xy_wind_ls = list(xy_wind_ls)
        Xy[row_idx, :] = np.array(xy_wind_ls)
        row_idx += 1
        #iterate from the second row to the bottom (deduct the last reference rows)
        for ri in range(wind_hei + 1, img_hei - wind_hei):
            #delete the first (2*wind_wid+1) elements, as it represents the first row of last window, add the corresponding window
            xy_wind_ls = xy_wind_ls[(2*wind_wid+1):] + list(img_arr[ri+wind_hei, (ci-wind_wid):(ci+wind_wid+1)])
            Xy[row_idx, :] = np.array(xy_wind_ls)
            row_idx += 1
    #Xy is now filled column by column
    n_col = Xy.shape[1]
    #Turn it into row by row and flatten it
    Xy = Xy.reshape((n_wid, n_hei, xy_dim)).transpose([1, 0, 2]).reshape((-1, xy_dim))
    #organize and obtain neighborhoods and the target for each pixel
    X, y = Xy[:, list(range(n_col // 2)) + list(range((n_col//2 + 1), n_col))], Xy[:, n_col//2]
    return X, y, n_hei, n_wid