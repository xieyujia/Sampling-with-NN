#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:41:12 2017

@author: yujia
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sess = tf.InteractiveSession()

#SETTINGS
N=1000
batchsize=100
nnode1=10
d=2
#input data generation
mu, sigma = np.zeros(2), np.eye(2)
data = np.random.multivariate_normal(mu, sigma, (N, 1))
data=np.squeeze(data)

#target function parameter
ap1=0.6
mup1=[5,3]
sigmap1=[[2,1],[1,3]]
ap2=0.4
mup2=[1,2]
sigmap2=[[2,1],[1,2]]

#plot 
def plot_all(data,ap1,mup1,sigmap1,ap2,mup2,sigmap2,i):

    plt.scatter(data[:,0],data[:,1],s=10)
        
    x = np.arange(-3, 11, 0.2)
    y = np.arange(-3, 8, 0.2)
    z=ap1*normpdf_figure(x,y,mup1,sigmap1)+ap2*normpdf_figure(x,y,mup2,sigmap2)
    C=plt.contour(x, y, z.T,linewidths=1.5)
    #plt.clabel(C, inline=1, fontsize=10)
    plt.title('Plot of  '+str(i))
    plt.figure(figsize=(20,20))
    plt.show()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)+1
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1)
  return tf.Variable(initial)
  
def next_batch(data,iter,batchsize):
  N=np.size(data,0)
  start=(iter*batchsize % N)
  return data[start:start+batchsize]

def normpdf(t,mu,sigma):
    a=np.linalg.inv(sigma)
    a= tf.cast(a,dtype=tf.float32)
    x=t-mu
    return tf.exp(-tf.reduce_sum(tf.matmul(x,a)* x /2,1))
    
def normpdf_figure(x,y,mu,sigma):
    a=np.linalg.inv(sigma)
    x=x-mu[0]
    y=y-mu[1]
    lenx=len(x)
    leny=len(y)
    z=np.zeros([lenx,leny])
    for i in range (0,lenx):
        for j in range (0,leny):
            z[i,j]=np.exp(-((x[i]*a[0,0]+y[j]*a[1,0])*x[i]+(x[i]*a[0,1]+y[j]*a[1,1])*y[j])/2)
    z=z/2/np.sqrt(np.pi*np.linalg.norm(sigma))
    return z
  
def source(x,mu,sigma):
    q=normpdf(x,mu,sigma)
    return q
      
def target(x,ap1,mup1,sigmap1,ap2,mup2,sigmap2):
    p=ap1*normpdf(x,mup1,sigmap1)+ap2*normpdf(x,mup2,sigmap2)
    return p
    
def kl(x,y,mu,sigma,det,ap1,mup1,sigmap1,ap2,mup2,sigmap2):
    element=tf.log(source(x,mu,sigma)/target(y,ap1,mup1,sigmap1,ap2,mup2,sigmap2)/det)
    det=target(y,ap1,mup1,sigmap1,ap2,mup2,sigmap2)

    return tf.reduce_sum(element, 0)

plot_all(data,ap1,mup1,sigmap1,ap2,mup2,sigmap2,'initial')
    
particle = tf.placeholder(tf.float32, shape=[None, d])
  
W_1 = weight_variable([d,nnode1])
b_1 = bias_variable([nnode1])

h_1 = tf.sigmoid(tf.matmul(particle,W_1) + b_1) #+0.01*particle



W_fc = weight_variable([nnode1,d])
b_fc = bias_variable([d])

#get result
T = tf.matmul(h_1,W_fc) + b_fc

    #loss function
    #loss = tf.contrib.distributions.kl(x,y)
temp=tf.exp(-tf.matmul(particle,W_1)-b_1)
temp1=temp/(1+temp)/(1+temp)
grad=0
for j in range(0,nnode1):
    grad=grad+tf.tensordot(tf.reshape(temp1[:,j],[batchsize,1]),tf.reshape(tf.matmul(tf.reshape(W_1[:,j],[d,1]),tf.reshape(W_fc[j,:],[1,d])),[1,d,d]),[[1], [0]])
    

det=tf.abs(tf.matrix_determinant(grad))

#det=tf.ones(batchsize)
loss =kl(particle,T,mu,sigma,det,ap1,mup1,sigmap1,ap2,mup2,sigmap2) 
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

clip_1 = W_1.assign(tf.clip_by_value(W_1, 0.001, W_1))
clip_fc = W_fc.assign(tf.clip_by_value(W_fc, 0.001, W_fc))

sess.run(tf.global_variables_initializer())
#train
for i in range(80001):
  batch = next_batch(data,i,batchsize)

  if i%1000 == 0:
    print("step %d"%(i))
    #print(W_1.eval())
    #print(W_fc.eval())
    
  if i%5000 ==0:
    train_result = T.eval(feed_dict={
        particle:data})
    #print(train_result)
    plot_all(train_result,ap1,mup1,sigmap1,ap2,mup2,sigmap2,"iter "+str(i))
  sess.run(
            [train_step, clip_1, clip_fc],
            feed_dict={particle:batch}
        )
  #train_step.run(feed_dict={x: batch})
    
  
#  
#  W_2=tf.abs(W_2)
#  W_fc=tf.abs(W_fc)
#  print(W_1.eval(),W_2.eval,W_fc.eval)
#put the restriction to W here in the future
  
#output figure
train_result = T.eval(feed_dict={
        particle:data})
plot_all(train_result,ap1,mup1,sigmap1,ap2,mup2,sigmap2,"iter"+str(i))

