#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:59:36 2017

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:26:00 2017

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:17:08 2017

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
lamda=10
#input data generation
mu, sigma = 0, 1
data = np.random.normal(mu, sigma, N)

#target function parameter
ap1=0.6
mup1=5
sigmap1=1
ap2=0.4
mup2=1
sigmap2=1

#plot 
def plot_all(data,ap1,mup1,sigmap1,ap2,mup2,sigmap2):
    bin=20
    N=len(data)
    h=plt.hist(data, bins=bin)
    figure_x=np.arange(-3, 7, 0.1)
    figure_y=ap1*normpdf_figure(figure_x,mup1,sigmap1)+ap2*normpdf_figure(figure_x,mup2,sigmap2)
    plt.plot(figure_x,figure_y*(h[1][1]-h[1][0])*N*1.2,'r',linewidth=2.0)
    plt.figure(figsize=(20,20))
    plt.show()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1)
  return tf.Variable(initial)
  
def next_batch(data,iter,batchsize):
  N=np.size(data,0)
  start=(iter*batchsize % N)
  return np.transpose([data[start:start+batchsize]])

def normpdf(t,mu,sigma):
    return tf.exp(-(t-mu)*(t-mu)/2/sigma)/2/tf.sqrt(np.pi*sigma)
    
def normpdf_figure(t,mu,sigma):
    return np.exp(-(t-mu)*(t-mu)/2/sigma)/2/np.sqrt(np.pi*sigma)
  
def source(x,mu,sigma):
    q=normpdf(x,mu,sigma)
    return q
      
def target(x,ap1,mup1,sigmap1,ap2,mup2,sigmap2):
    p=ap1*normpdf(x,mup1,sigmap1)+ap2*normpdf(x,mup2,sigmap2)
    return p
    
def kl(x,y,mu,sigma,det,ap1,mup1,sigmap1,ap2,mup2,sigmap2):
    element=tf.log(source(x,mu,sigma)/target(y,ap1,mup1,sigmap1,ap2,mup2,sigmap2)/det)
    return tf.reduce_sum(element, 0)

plot_all(data,ap1,mup1,sigmap1,ap2,mup2,sigmap2)
    
x = tf.placeholder(tf.float32, shape=[None, 1])
  
W_1 = weight_variable([1,nnode1])
b_1 = bias_variable([nnode1])

h_1 = tf.sigmoid(tf.matmul(x,W_1) + b_1) #+0.01*x



W_fc = weight_variable([nnode1,1])
b_fc = bias_variable([1])

#get result
#y = np.asarray(W_fc).dot(h_drop) + b_fc
y = tf.matmul(h_1,W_fc) + b_fc

#loss function
#loss = tf.contrib.distributions.kl(x,y)
temp=tf.exp(-tf.matmul(x,W_1)-b_1)
grad1=temp/(1+temp)/(1+temp)*W_1

grad=tf.matmul(grad1,W_fc)
#det=tf.matrix_determinant(grad)
det=grad

loss =kl(x,y,mu,sigma,det,ap1,mup1,sigmap1,ap2,mup2,sigmap2) #-lamda*(tf.reduce_sum( W_1)+tf.reduce_sum( W_fc))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

clip_1 = W_1.assign(tf.clip_by_value(W_1, 0.1, W_1))
clip_fc = W_fc.assign(tf.clip_by_value(W_fc, 0.1, W_fc))

sess.run(tf.global_variables_initializer())
#train
for i in range(80000):
  batch = next_batch(data,i,batchsize)

  if i%1000 == 0:
    print("step %d"%(i))
    print(W_1.eval())
    print(W_fc.eval())
    
  if i%5000 ==0:
    train_result = y.eval(feed_dict={
        x:np.transpose([data])})
    #print(train_result)
    plot_all(train_result,ap1,mup1,sigmap1,ap2,mup2,sigmap2)
  sess.run(
            [train_step, clip_1, clip_fc],
            feed_dict={x: batch}
        )
  #train_step.run(feed_dict={x: batch})
    
  
#  
#  W_2=tf.abs(W_2)
#  W_fc=tf.abs(W_fc)
#  print(W_1.eval(),W_2.eval,W_fc.eval)
#put the restriction to W here in the future
  
#output figure
train_result = y.eval(feed_dict={
        x:np.transpose([data])})
plot_all(train_result,ap1,mup1,sigmap1,ap2,mup2,sigmap2)

