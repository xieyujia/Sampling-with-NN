#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:44:18 2017

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:41:12 2017

@author: yujia
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
sess = tf.InteractiveSession()

#SETTINGS
N=1000
batchsize=100
nnode1=16
d=2
#input data generation
mu, sigma = np.zeros(d), np.eye(d)
data = np.random.multivariate_normal(mu, sigma, (N, 1))
data=np.squeeze(data)

#target function parameter
ap1=0.6
mup1=[5,3]
sigmap1=[[2,1],[1,3]]
ap2=0.4
mup2=[1,2]
sigmap2=[[2,1],[1,2]]

#ap1=0.6
#mup1=[5,3,1]
#sigmap1=[[2,1,1],[1,3,1],[1,1,4]]
#ap2=0.4
#mup2=[1,2,1]
#sigmap2=[[2,1,1],[1,3,1],[1,1,4]]

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
  initial = tf.truncated_normal(shape, stddev=0.1)+0.5
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1)
  return tf.Variable(initial)
  
def next_batch(data,iter,batchsize):
  N=np.size(data,0)
  start=(iter*batchsize % N)
  return data[start:start+batchsize]

def multiply(mylist):
    product=1
    for i in mylist:
        product=product*i
    return product

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

# Get pairs of indices of positions
indices = list(zip(*np.tril_indices(d)))
indices = tf.constant([list(i) for i in indices], dtype=tf.int64)


# Test values to load into matrix
W_1_real = weight_variable([int(d*(d+1)/2),nnode1])

W_1 = [tf.SparseTensor(indices=indices, values=W_1_real[:,i], dense_shape=[d, d]) for i in range(nnode1)]
b_1 = [bias_variable([d]) for i in range(nnode1)]

#W_1_temp=[tf.sparse_tensor_to_dense(W_1[i], 0.0) for i in range(nnode1)]
h_1 =[ tf.sigmoid(tf.transpose(tf.sparse_tensor_dense_matmul(W_1[i], tf.transpose(particle)) ) + b_1[i]) for i in range(nnode1)]
#h_1 = [tf.sigmoid(tf.matmul(particle,W_1_temp[i]) + b_1[i]) for i in range(nnode1)]


W_fc_real = weight_variable([nnode1, int(d*(d+1)/2)])
W_fc = [tf.SparseTensor(indices=indices, values=W_fc_real[i,:], dense_shape=[d, d]) for i in range(nnode1)]
b_fc = bias_variable([d])
#W_fc_temp=[ tf.sparse_tensor_to_dense(W_fc[i], 0.0) for i in range(nnode1)]
#get result
T = sum([tf.transpose(tf.sparse_tensor_dense_matmul(W_fc[i], tf.transpose(h_1[i])) ) for i in range(nnode1)]) + b_fc
#T = sum([tf.matmul(h_1[i],W_fc_temp[i] ) for i in range(nnode1)]) + b_fc
#print(T.get_shape())



temp=[tf.exp(-tf.transpose(tf.sparse_tensor_dense_matmul(W_1[i], tf.transpose(particle)) ) - b_1[i]) for i in range(nnode1)]
#temp=[tf.exp( -tf.matmul(particle,W_1_temp[i]) - b_1[i]) for i in range(nnode1)]
sig1=[temp[i]/(1+temp[i])/(1+temp[i]) for i in range(nnode1)]
      
#product=[tf.matmul(tf.sparse_tensor_to_dense(W_fc[i], 0.0),
#              tf.sparse_tensor_to_dense(W_1[i], 0.0),
#              transpose_a=True, transpose_b=True,
#              a_is_sparse=True, b_is_sparse=True) for i in range(nnode1)]
                   
#W_1_temp=[ tf.transpose(tf.sparse_tensor_to_dense(W_1[i], 0.0)) for i in range(nnode1)]
#W_fc_temp=[ tf.transpose (tf.sparse_tensor_to_dense(W_fc[i], 0.0)) for i in range(nnode1)]

#grad=0
#for k in range(0,d):
#    for i in range(nnode1):
#        grad=grad+tf.tensordot(tf.reshape(sig1[i][:,k],[batchsize,1]),tf.reshape(tf.matmul(tf.reshape(W_1_temp[i][:,k],[d,1]),tf.reshape(W_fc_temp[i][k,:],[1,d])),[1,d,d]),[[1], [0]])
#


det1=[0  for l in range (d)]
for l in range(0,d):
        for i in range(nnode1):
            #det1[l] = det1[l]+W_fc_temp[i][k,l]*W_1_temp[i][l,k]*sig1[i][:,k]
            det1[l] = det1[l]+W_fc[i].values[int(d*l-(l*l+l)/2+l)]*W_1[i].values[int(d*l-(l*l+l)/2+l)]*sig1[i][:,l]
           
#print(np.shape(det))
#det2= [sum(det1[i]) for i in range (d)]
det=multiply(det1)
#print(det.get_shape())

loss =kl(particle,T,mu,sigma,det,ap1,mup1,sigmap1,ap2,mup2,sigmap2) 
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

clip_1 = [W_1[i].values.assign(tf.clip_by_value(W_1[i].values, 0.001, W_1[i].values)) for i in range(nnode1)]
clip_fc =[ W_fc[i].values.assign(tf.clip_by_value(W_fc[i].values, 0.001, W_fc[i].values)) for i in range(nnode1)]

sess.run(tf.global_variables_initializer())
#train

start_time = time.time()
for i in range(80000):
  batch = next_batch(data,i,batchsize)
  
  if i%1000 == 0:
    print("step %d"%(i))
    print(time.time()-start_time)
#    print(sess.run(det, feed_dict={
#        particle:batch}))
    print(W_1[0].eval())
    print(W_fc[0].eval())
    
  if i%5000 ==0:
    train_result = T.eval(feed_dict={
        particle:data})
    #print(train_result)
    plot_all(train_result,ap1,mup1,sigmap1,ap2,mup2,sigmap2,"iter "+str(i))
  sess.run(
            [train_step, clip_1, clip_fc],
            feed_dict={particle:batch}
        )
#  W_1_monitor=sess.run( W_1_real)
#  W_fc_monitor=sess.run( W_fc_real)
#  print(W_1_monitor,W_fc_monitor)
#  det_monitor=sess.run(det, feed_dict={particle: batch})
#  print(det_monitor)
  #train_step.run(feed_dict={x: batch})
    
  

#output figure
train_result = T.eval(feed_dict={
        particle:data})
plot_all(train_result,ap1,mup1,sigmap1,ap2,mup2,sigmap2,"iter"+str(i))

