#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:23:09 2017

@author: yujia
"""

import numpy as np
import numpy.matlib

def evaluation(particle, x_test, y_test):
    #calculate the prediction error and log-likelihood
    #theta: M * d, logistic regression weights
    #X_test:  N0 * d, input data
    #y_test:  N0 * 1, contains the label (+1/-1)
    
    (M,D)=np.shape(particle)
    particle = particle[:,0:D-1]  # only need w to evaluate accuracy and likelihood

    n_test = len(y_test) # number of evaluation data points
    y_test=y_test*2-1
    prob = np.zeros([n_test, M])
    for t in range (0,M):
        prob[:, t] = 1 / (1 + np.exp( y_test* np.sum(-np.matlib.repmat(particle[t,:], n_test, 1)* x_test, axis=1)))

    prob = np.mean(prob, 1)
    acc = np.mean(prob > 0.5)
    llh = np.mean(np.log(prob))
    return (acc,llh)