#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:28:43 2017

@author: yujia
"""

#SETTINGS
M=20
max_iter=100000
nnode1=10
split_ratio=0.8

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA 
import time
#from tensorflow.examples.tutorials.mnist import input_data
import veriT_core
import evaluate

#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
print("Downloading training data ...")
mnist = fetch_mldata('MNIST original')

print("Processing training data ...")
X = mnist.data
y = mnist.target

#priginal data
X0=X[y==0,:]
X1 = X[y==1,:]

#split data
data_train = np.concatenate((X0[0:round(len(X0)*split_ratio),:],X1[0:round(len(X1)*split_ratio),:]), axis=0)
data_test = np.concatenate((X0[round(len(X0)*split_ratio):round(len(X0)),:],X1[round(len(X1)*split_ratio):round(len(X1)),:]), axis=0)
datay_train=np.concatenate((np.ones(round(len(X0)*split_ratio))*0, np.ones(round(len(X1)*split_ratio))*1),axis=0)
datay_test=np.concatenate((np.ones(round(len(X0)*(1-split_ratio)))*0, np.ones(round(len(X1)*(1-split_ratio)))*1),axis=0)
     
#dimension reduction
pca=PCA(n_components=3, copy=True, whiten=False)
data_train=pca.fit_transform(data_train)
data_test=pca.transform(data_test)

#data with bias absorbed
data_train = np.concatenate((data_train,np.expand_dims(np.ones(np.size(data_train,0)),axis=1)),axis=1)
data_test = np.concatenate((data_test,np.expand_dims(np.ones(np.size(data_test,0)),axis=1)),axis=1)

(N,D)=np.shape(data_train)

#input data generation
D=D+1
particle = np.random.multivariate_normal(np.ones(D), np.eye(D), M)

#datay_train=np.expand_dims(datay_train,axis=1)
#print(np.shape(datay_train))
#datay_train=np.matlib.repmat(datay_train, 1, M)
##datay=np.matlib.repmat(datay, 1, M)
#print(np.shape(datay_train))


#train
start_time = time.time()
(result,w1,w2)=veriT_core.core(particle,max_iter,nnode1,data_train, datay_train, data_test, datay_test)
end_time = time.time()

#evaluation
print("Evaluating the results ...")
[acc, llh] = evaluate.evaluation(result, data_test, datay_test)
print('Result of variT: testing accuracy:', acc,', testing loglikelihood: ',llh,', running time: ' , end_time-start_time);








