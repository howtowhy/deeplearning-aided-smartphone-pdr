

 # classification

'''
    filename: main.py

    main.py : main script

    author: Seoyeon Yang
    date  : 2018 July
    references: https://github.com/motlabs/mot-dev/blob/180506_tfdata_jhlee/lab11_tfdata_example/data_manager%20(mnist).ipynb
                https://github.com/jwkanggist/tensorflowlite
                https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/08__queues_threads
                https://nbviewer.jupyter.org/github/aisolab/CS20/blob/master/Lec11_Recurrent%20Neural%20Networks/Lec11_Sequence%20Classificaion%20by%20LSTM.ipynb
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
import matplotlib.pyplot as plt
from scipy import io
import data_loader2
import hyper_parameter
from layer import conv_layer, max_pool_2x2, full_layer, max_pool_3x3
import lstm
import time
from datetime import datetime
import random
# tf record 만들어졌는지 확인하는 전역변수만들것

def main():

    ## ---------------------------------------------------------------------------------------------------------------------
    ## hyper parameter

    class_num = 6
    batch_size = 100
    height_ = 23100
    width_ = 5

    ## ---------------------------------------------------------------------------------------------------------------------
    ## Save dir
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    board_logdir = "C:\\Users\\SNUGL\\tensorboard\\"
    ckpt_logdir = "D:\dev\jejucamp-seoyeon\classification\\model_ckpt\\"

    ## ---------------------------------------------------------------------------------------------------------------------
    ## data checking

    check = 1  # 데이터 저장 안됬을 시 0, 데이터 저장 됬을 시 1
    check2 = 1 # ckpt 로딩 안할시 0 , ckpt 로딩할 시 1
    ## ---------------------------------------------------------------------------------------------------------------------
    ## data loader

    if check == 0 :
    # training data
        train_parser = data_loader2.parse_cambridge() # class
    # name list, label list, path list parser
        train_name_list, train_label_list, train_path_list = train_parser.label_extract('train')
    # read the data and concatinate
        train_data, train_label = train_parser.loading_data(train_path_list, height_, width_, None, 'train')
        np.save('train_data_2', train_data)
        np.save('train_label_2', train_label)
    # test data
        test_parser = data_loader2.parse_cambridge()
    # name list, label list, path list parser
        test_name_list, test_label_list, test_path_list = test_parser.label_extract('test')
    # read the data and concatinate
        test_data, test_label = test_parser.loading_data(test_path_list, height_, width_, None, 'test')
        np.save('test_data_2', test_data)
        np.save('test_label_2', test_label)
        check = 2

    ''' if check == 1 :
        train_data = np.load('train_data_2.npy')
        train_label = np.load('train_label_2.npy')
        test_data = np.load('test_data_2.npy')
        test_label = np.load('test_label_2.npy')
        check = 2'''

    if check == 1:
        mat_file = io.loadmat('real_lasting.mat')
        data = mat_file['real_data']
        b = np.transpose(data)
        train_data = b[:1001]
        test_data = b[1001:]
        lab = mat_file['real_label']
        c = np.transpose(lab)
        train_label = c[:1001]
        test_label = c[1001:]
    ## ---------------------------------------------------------------------------------------------------------------------
    ## check point loading

    ## ---------------------------------------------------------------------------------------------------------------------
    ## config env


    # training constant config

    epochs = 1000
    iter_ = 1
    lr = .0001
#e-5
    num_end  = []
    for w in range(batch_size) :
        num_end.append(height_)
    total_sample = np.shape(train_data)[0]
    total_batch = int(np.shape(train_data)[0] / batch_size)
    print(epochs)

    ## ---------------------------------------------------------------------------------------------------------------------
    ## model defined

    jeju_graph = tf.Graph()

    with jeju_graph.as_default():

        lstm_model = lstm.LstmModel()
        # input : A 'batch_size' x 'max_frames' x 'num_features'
        # max_frames : time series,
        # num_feature : contributes

        x = tf.placeholder(dtype=tf.float32,
                       shape=[None, height_, width_],
                       name='input')
        y = tf.placeholder(dtype=tf.int64,
                        shape=[None, class_num],
                        name='output')

        with tf.name_scope('output'):
            output = lstm_model.create_model(x, class_num, num_end)
            tf.summary.histogram('output',output)

        # create training op
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        with tf.name_scope('loss_l'):
            loss_l = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output) # cross_entroy : classification , l2_loss : regression
            tf.summary.scalar('loss_l',loss_l)
        training_op = opt.minimize(loss=loss_l)

        # prediction
        pred = tf.argmax(output, 1)
        pred_y = tf.argmax(y, 1)
        diff = tf.equal(pred, pred_y)
        hey = tf.cast(diff, tf.float32)
        predict_accuracy = tf.reduce_mean(hey)
        with tf.name_scope('accuracy'):
            tf.summary.scalar('accuracy',predict_accuracy)

        merged = tf.summary.merge_all()

    ## ---------------------------------------------------------------------------------------------------------------------
    ## savor

        tf_saver = tf.train.Saver() #max_to_keep=7, keep_checkpoint_every_n_hours=1

    ## ---------------------------------------------------------------------------------------------------------------------
    ## Tensorboard

    train_writer = tf.summary.FileWriter(logdir=board_logdir + '/train', graph = jeju_graph)
    test_writer = tf.summary.FileWriter(logdir=board_logdir + '/test', graph=jeju_graph)
    ## ---------------------------------------------------------------------------------------------------------------------
    # session
    with tf.Session(graph=jeju_graph) as sess:
        if check2 == 1 :
            tf_saver.restore(sess, os.path.join("D:\\dev\\jejucamp-seoyeon\\classification\\model_ckpt","ckpt-216310"))
        sess.run(tf.global_variables_initializer())

        # prepare session
        tr_loss_hist = []
        global_step = 0
        save_accuracy = []
        test_len = test_data.shape[0]
        train_len = train_data.shape[0]
        display_num = 0
        sum_accuracy = 0

        for e in range(epochs) :

            for i in range(total_batch) :
                random_idex = random.randrange(0, test_len - batch_size)
                batch_validation_data = test_data[random_idex:random_idex + batch_size, ...]
                batch_validation_label = test_label[random_idex:random_idex + batch_size, ...]
                data_start_index = i * batch_size
                data_end_index = (i + 1) * batch_size

                batch_train_data = train_data[data_start_index:data_end_index, ...]
                batch_train_label = train_label[data_start_index:data_end_index]

                avg_tr_loss = 0
                tr_step = 0
                try:
                    for h in range(iter_) : # for 문으로 tr 고정
                        _, tr_loss, summary1 = sess.run(fetches=[training_op, loss_l, merged], feed_dict = {x: batch_train_data,
                                 y: batch_train_label})
                        #print('epoch : {:3}, batch_step : {:3}, tr_step : {:3}, tr_loss : {:.5f}'.format(e + 1, i +1, tr_step + 1, tr_loss))
                        avg_tr_loss += tr_loss
                        tr_step += 1
                        global_step += 1

                except tf.errors.OutOfRangeError:
                    pass

                avg_tr_loss /= tr_step
                tr_loss_hist.append(avg_tr_loss)
                acc = predict_accuracy.eval(feed_dict={x: batch_validation_data,
                                                        y: batch_validation_label}, session=sess)
                summary2 = merged.eval(feed_dict={x: batch_validation_data,
                                                        y: batch_validation_label}, session=sess)
                display_num = display_num + 1

                sum_accuracy = sum_accuracy + acc
                if(display_num%10==0):
                    avg_save_accuracy = sum_accuracy/display_num
                    print('epoch : {:3}, batch_step : {:3}/{:3}, avg_tr_loss : {:.5f}, prediction : {:.5f}'.format(e + 1,
                                                                                                               i + 1,
                                                                                                               total_batch,
                                                                                                               avg_tr_loss,
                                                                                                               acc))

                    tf_saver.save(sess, os.path.join(ckpt_logdir,"ckpt"), global_step=global_step)
                    display_num = 0
                    sum_accuracy = 0
                    save_accuracy.append(avg_save_accuracy)
                    train_writer.add_summary(summary1, global_step)
                    test_writer.add_summary(summary2, global_step)




    ## ---------------------------------------------------------------------------------------------------------------------
    ## plotting

    ''' plt.plot(tr_loss_hist, label='train')
    plt.show()
    plt.plot(save_accuracy, label='accuracy')
    plt.show() '''


if __name__ == "__main__":
     main()


