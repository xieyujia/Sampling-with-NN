#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:38:27 2017

@author: yujia
"""
import numpy as np
import tensorflow as tf



def normpdf(t,mu,sigma):
    a=np.linalg.inv(sigma)
    a= tf.cast(a,dtype=tf.float32)
    x=t-mu
    return tf.exp(-tf.reduce_sum(tf.matmul(x,a)* x /2,1))

def targetp(particle, X_train,y_train):
    
#
    (M,D)=np.shape(particle)

    w=particle[:,0:D-1]     #logistic weights
    alpha= tf.exp(particle[:,D-1])   #the last column is logalpha

    wt=tf.matmul(X_train,tf.transpose(w))

    llh=y_train*wt-tf.log(1+tf.exp(wt))
#    print(llh.get_shape())
    w_prior = - alpha * tf.reduce_sum(w*w,1)
    logp = tf.reduce_sum(llh,0)  + w_prior
#    print(logp.get_shape())
    return logp 

    
    
    
    
def targetp_test(particle, X_train,y_train):
    (M,D)=np.shape(particle)
    w=particle[:,0:D-1] 

#    X_train0=X_train[y_train==0,:]
#    X_train1=X_train[y_train==1,:]
#
#    mu0=np.average(X_train0, axis=0)
#    mu1=np.average(X_train1, axis=0)
#
#    X=np.concatenate((X_train0-mu0, X_train1-mu1), axis=0)
#    sigma=X.T.dot(X)/len(X)
#
#    sigma_inv=np.linalg.inv(sigma)
#    wt=sigma_inv.dot(mu1-mu0)
#    b=mu0.dot(sigma_inv).dot(mu0.T)/2 - mu1.dot(sigma_inv).dot(mu1.T)/2 + np.log(len(X_train1)/len(X_train0))
#
#    theta=np.concatenate((wt, [b]), axis=0)
#    theta=np.reshape(theta,[1,D-1])
    
    mu=[-1.730642696389892571e-02	,-8.695146556670009953e-04,	1.148529568187250200e-03, 1.232320870136672575e+00]
    #mu=[-1.704876741637844154e-02, 1.213968244358172743e+00]
    #sigma=[[2,1],[1,3]]
    sigma=[[2,1,0,1],[1,3,0,1],[0,0,4,1],[1,1,1,1]]

    p=normpdf(w,mu,sigma)
    logp=tf.log(p)
    return logp  
