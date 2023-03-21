#Load the model
from numpy import *
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#changed
tf.random.set_seed(0)
#changed
tf.compat.v1.reset_default_graph()

train_x = zeros((50, 227,227,3)).astype(float32)
train_y = zeros((1, 180))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read the Labels

labels=open("./content/labels.txt").read().splitlines()

################################################################################

net_data = load(open("./content/alexnet180_tf.npy", "rb"), encoding="latin1",allow_pickle=True).item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

#changed
#desactiva la ejecución ansiosa
tf.compat.v1.disable_eager_execution()
#changed
x = tf.compat.v1.placeholder(tf.float32, (None,) + xdim)


radius = 2; alpha = 0.00002; beta = 0.75; bias = 1.0
lrn0 = tf.nn.local_response_normalization(x,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"]['weights'])
conv1b = tf.Variable(net_data["conv1"]['biases'])
conv1_in = conv(lrn0, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1

lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"]['weights'])
conv2b = tf.Variable(net_data["conv2"]['biases'])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"]['weights'])
conv3b = tf.Variable(net_data["conv3"]['biases'])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)
#lrn3=tf.nn.batch_normalization(conv3, tf.Variable(net_data["bn3"]["mean"]), tf.Variable(net_data["bn3"]["variance"]), offset, scale, variance_epsilon)
#lrn3=batch_norm_layer(conv3)
#radius = 1; alpha = 1.0; beta = 0.980000019073; bias = 9.99999974738e-05
lrn3 = tf.nn.local_response_normalization(conv3,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"]['weights'])
conv4b = tf.Variable(net_data["conv4"]['biases'])
conv4_in = conv(lrn3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

lrn4 = tf.nn.local_response_normalization(conv4,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"]['weights'])
conv5b = tf.Variable(net_data["conv5"]['biases'])
conv5_in = conv(lrn4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"]['weights'])
fc6b = tf.Variable(net_data["fc6"]['biases'])
#changed fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
fc6 = tf.compat.v1.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"]['weights'])
fc7b = tf.Variable(net_data["fc7"]['biases'])
#changed fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
fc7 = tf.compat.v1.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc82"]['weights'])
fc8b = tf.Variable(net_data["fc82"]['biases'])
#changed fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
fc8 = tf.compat.v1.nn.xw_plus_b(fc7, fc8W, fc8b)

#prob
#softmax(name='prob'))

prob = tf.nn.softmax(fc8)

#prob= tf.layers.dense(input=fc8,  tf.nn.softmax)
#changed init = tf.initialize_all_variables()
init = tf.compat.v1.initialize_all_variables()
#changed sess = tf.Session()
sess = tf.compat.v1.Session()
sess.run(init)
mu="./content/mean.npy"   #mean npy
mu=np.load(mu)
mu.resize((3,227,227))
mu=mu.mean(1).mean(1)
print("\n\n\nModel Ready for Testing...")

#test
import cv2

ims=["./content/cam.jpg","./content/orange1_B_01.png","./content/cup1_B_01.png"]
for i in ims:
  im = cv2.imread(i)
  im2= cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  im1=cv2.resize(im,(227,227),cv2.INTER_CUBIC)
  im1=np.array(im1,np.float32)
  im1 -=  mu
  plt.imshow(im2.astype(uint8))
  plt.show()


  output = sess.run(prob, feed_dict = {x:[im1]})[0]

  top5=output.argsort()[-5:][::-1]
  top1c=labels[top5[0]].split("/")[1]
  top1p=labels[top5[0]].split("/")[0]

  top=[]
  acc=[]
  c=[]

  for i in top5:
      l=labels[i]
      acc.append(output[i])
      top.append(l.split("/")[0])

  c=[top.count(i) for i in set(top)]
  if top.count(top1p) == max(c):
    top=top1p
  else :
    top=max(set(top),key=top.count)


  print( "\n\n\nThe Top 1 parent category is: ",top1p, "with accuracy:",acc[0],"\nThe Top 1 child category is: ",top1c, "with accuracy:",acc[0],"\nThe Top 5 majority parent is:", top)
  print("top5 categories: ",[labels[i] for i in top5])