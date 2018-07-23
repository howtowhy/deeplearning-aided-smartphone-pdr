

 # classification

'''
    filename: main.py

    main.py : main script

    author: Seoyeon Yang
    date  : 2018 July
    references: https://github.com/motlabs/mot-dev/blob/180506_tfdata_jhlee/lab11_tfdata_example/data_manager%20(mnist).ipynb
                https://github.com/jwkanggist/tensorflowlite
                https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/08__queues_threads
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


import gzip
import sys
import os
from os import getcwd
import numpy as np
from six.moves import urllib
from glob import glob
from PIL import Image

import data_loader
import hyper_parameter
from layer import conv_layer, max_pool_2x2, full_layer

# tf record 만들어졌는지 확인하는 전역변수만들것

def main():
    check =0

    # main fuction
    if check == 0 :
    # training data
        train_parser = data_loader.parse_cambridge() # class
    # name list, label list, path list parser
        train_name_list, train_label_list, train_path_list = train_parser.label_extract('train')
    # read the data and concatinate
        train_parser.writing_data_to_tfrecord(train_path_list, 6 ,'train')

    # test data
        test_parser = data_loader.parse_cambridge()
    # name list, label list, path list parser
        test_name_list, test_label_list, test_path_list = test_parser.label_extract('test')
    # read the data and concatinate
        test_parser.writing_data_to_tfrecord(test_path_list, 6, 'test')

    check = 1

    #re-read and parsing the data

    batch_size = 5
    epoch_size = 10

    read_train_parser = data_loader.parse_cambridge()
    train_iterator = read_train_parser.read_data_from_tfrecode('train', batch_size, epoch_size, True) #, train_name
    train_next_element = train_iterator.get_next()


    read_test_parser = data_loader.parse_cambridge()
    test_iterator = read_test_parser.read_data_from_tfrecode('test', batch_size, epoch_size, False)  # , test_name
    test_next_element = test_iterator.get_next()

    #read_train_parser.preprocessing(train_image, train_label, train_channel, train_sample)

    #input_image = train_image
    #output_image = train_label

    #STEPS = 1000
    #MINIBATCH_SIZE = 10

    x = tf.placeholder(dtype=tf.float32,
                                      shape=[None, 60000],
                                      name='input')
    y_ = tf.placeholder(dtype=tf.int64,
                                      shape=[None, 1, 6],
                                      name='output')

    x_image = tf.reshape(x,[-1,1000,10,6],name='image_end')
   # batch_ind = tf.placeholder(dtype=tf.int64, shape =[1])

    '''    conv1 = conv_layer(x_image, shape=[3, 3, 6, 32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[3, 3, 32, 64])
    conv2_pool = max_pool_2x2(conv2)

    conv2_flat = tf.reshape(conv2_pool, [-1, 5 * 500 * 64])
    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

    keep_prob = tf.placeholder(tf.float32)
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, 6)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) '''


    with tf.Session() as sess:
       sess.run(train_iterator.initializer)
       image = sess.run(train_next_element)
       print(image)
    return 0

if __name__ == "__main__":
     main()
