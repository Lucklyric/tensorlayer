"""
Created on 2017-09-14
Project: tensorlayer
File: tutorial_plot_net
...
@author: Alvin(Xinyao) Sun
"""
import tensorlayer as tl
import tensorflow as tf

x_1 = tf.placeholder(tf.float32, [None, 1], 'x_1')
x_2 = tf.placeholder(tf.float32, [None, 10], 'x_2')
lin = tl.layers.InputLayer(x_1, 'Input_l')
rin = tl.layers.InputLayer(x_2, 'Input_r')
l1 = tl.layers.DenseLayer(lin, name='F1')
l2 = tl.layers.DenseLayer(l1, name='F2')
l3 = tl.layers.DenseLayer(l2, name='F3')
r1 = tl.layers.DenseLayer(rin, name='Fr1')
r1 = tl.layers.DropoutLayer(r1, name='Drop')
l5 = tl.layers.ConcatLayer([lin, rin, l1, l2, l3, r1], name='C')

tl.visualize.plot_net(l5, show_shapes=True)
