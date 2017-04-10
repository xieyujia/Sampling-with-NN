#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:51:43 2017

@author: yujia
"""
import numpy as np
import tensorflow as tf

import target
import evaluate

def core(data ,max_iter,nnode1,datax, datay,data_test, datay_test):


#SETTINGS
    (M,d)=np.shape(data)
    N=len(datay)
    batchsize=np.minimum(20,M)

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)+1
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.truncated_normal(shape, stddev=1)
      return tf.Variable(initial)
      
    def next_batch(data,iter,batchsize):
      N=np.size(data,0)
      start=(iter*batchsize % N)
      return data[start:start+batchsize,:]
    
    def source(t):
        q=tf.exp(-tf.reduce_sum(t*t,1))
        return q
        
    def kl(particle,T,det,X_train,y_train):
        element=tf.log(source(particle)/det)-target.targetp_test(T,X_train,y_train)
        return tf.reduce_sum(element, 0)
    
    #process the real data
    rand_index = np.random.permutation(N)
    datax=datax[rand_index,:]
    datay=datay[rand_index,]
    datay=np.expand_dims(datay,axis=1)
    datay=np.repeat(datay, batchsize, axis=1)
        
    print("Initializing the graph ...")
    particle = tf.placeholder(tf.float32, shape=[None, d])
    X_train = tf.placeholder(tf.float32, shape=[None, d-1])
    y_train = tf.placeholder(tf.float32, shape=[None, batchsize])
      
    W_1 = weight_variable([d,nnode1])
    b_1 = bias_variable([nnode1])
    clip_1 = W_1.assign(tf.clip_by_value(W_1, 0.01, W_1))
    h_1 = tf.sigmoid(tf.matmul(particle,W_1) + b_1) #+0.01*particle
    
    
    
    W_fc = weight_variable([nnode1,d])
    b_fc = bias_variable([d])
    clip_fc = W_fc.assign(tf.clip_by_value(W_fc, 0.1, W_fc))
    #get result
    T = tf.matmul(h_1,W_fc) + b_fc
    
    #loss function
    #loss = tf.contrib.distributions.kl(x,y)
    temp=tf.exp(-tf.matmul(particle,W_1)-b_1)
    temp1=temp/(1+temp)/(1+temp)
    grad=0
    for j in range(0,nnode1):
        grad=grad+tf.tensordot(tf.reshape(temp1[:,j],[batchsize,1]),tf.reshape(tf.matmul(tf.reshape(W_1[:,j],[d,1]),tf.reshape(W_fc[j,:],[1,d])),[1,d,d]),[[1], [0]])
        
    if (d==1):
        det=tf.abs(grad)
    else:
        det=tf.abs(tf.matrix_determinant(grad))
    #det=tf.ones(batchsize)
    loss =kl(particle,T,det,X_train,y_train) 
    #loss =tf.losses.absolute_difference(T,particle+1)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    #clip_1 = W_1.assign(tf.clip_by_value(W_1, 0.1, W_1))
    #clip_fc = W_fc.assign(tf.clip_by_value(W_fc, 0.1, W_fc))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #train
    print("Start training ...")
    for i in range(max_iter):

      batch_realx=next_batch(datax,i,batchsize*2)
      batch_realy=next_batch(datay,i,batchsize*2)
      batch = next_batch(data,i,batchsize)
#      det_monitor=sess.run(det, feed_dict={particle: batch,X_train: batch_realx,y_train: batch_realy})
#      print(det_monitor)
      sess.run(
                [ clip_1, clip_fc, train_step],
                feed_dict={particle: batch,X_train: batch_realx,y_train: batch_realy}
              )
      if i%1000 == 1:
        print("step %d"%(i))
      if i%5000 == 1:
        [acc, llh] = evaluate.evaluation(result, data_test, datay_test)
        print('Result of variT: testing accuracy:', acc,', testing loglikelihood: ',llh);


        #print(W_1.eval())
        #print(W_fc.eval())
      result,w1,w2=sess.run(
                [T,W_1,W_fc],
                feed_dict={particle: batch,X_train: batch_realx,y_train: batch_realy}
              )
    return (result,w1,w2)
        
    
    
