

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
        train_parser.writing_data_to_tfrecord(train_name_list, train_path_list, train_label_list,'train')

    # test data
        test_parser = data_loader.parse_cambridge()
    # name list, label list, path list parser
        test_name_list, test_label_list, test_path_list = test_parser.label_extract('test')
    # read the data and concatinate
        test_parser.writing_data_to_tfrecord(test_name_list, test_path_list, test_label_list, 'test')
        check = 1

    #re-read and parsing the data

    read_train_parser = data_loader.parse_cambridge()
    train_image, train_label, train_channel, train_sample = read_train_parser.read_data_from_tfrecode('train') #, train_name

    read_test_parser = data_loader.parse_cambridge()
    test_image, test_label, test_channel, test_sample = read_test_parser.read_data_from_tfrecode('test')  # , test_name
    #read_train_parser.preprocessing(train_image, train_label, train_channel, train_sample)

    input_image = train_image
    output_image = train_label

    STEPS = 1000
    MINIBATCH_SIZE = 10

    '''x = tf.placeholder(dtype=tf.float32,
                                      shape=[None, 1000,
                                             10,
                                             6],
                                      name='input')
    y_ = tf.placeholder(dtype=tf.int64,
                                      shape=[None, ],
                                      name='output')

    x_image = x'''
   # batch_ind = tf.placeholder(dtype=tf.int64, shape =[1])

    conv1 = conv_layer(input_image, shape=[5, 5, 6, 32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)

    conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])
    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))


    full1_drop = tf.nn.dropout(full_1, keep_prob=1)

    y_conv = full_layer(full1_drop, 10)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=output_image))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(output_image, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ind = 0
        bef_batch_ind = 0
        for i in range(STEPS):
            #bef_batch_ind = batch_ind
            #feed_dict = {batch_ind:ind+5}
            input_image = train_image
            output_image = train_label
           # input_image = train_image[bef_batch_ind:batch_ind]
           # output_image = train_label[bef_batch_ind:batch_ind]
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy)
                #train_accuracy = sess.run()
                print("step {}, training accuracy {}".format(i, train_accuracy))

            sess.run(train_step)

        X = test_image
        Y = test_label

        test_accuracy = np.mean(
            [sess.run(accuracy) for i in range(10)])

    print("test accuracy: {}".format(test_accuracy))
    return 0

    # parsing
'''
    tar_name = "p1.1_Female_20-29_170-179cm_Hand_held" # 이거를 list 로 대입 가능하게
        tar_label = 1 #label list랑 같이 넘겨주자
        path = "D:\\dev\\jejucamp-seoyeon\\classification\\ubcomp13\\"+str(tar_label)+"\\"+tar_name+".out"
        parser.parsing_one(tar_name,path)
'''




if __name__ == "__main__":
     main()
