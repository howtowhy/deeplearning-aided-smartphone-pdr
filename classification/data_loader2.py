

 # classification

'''
    filename: data_loader.py

    data_loader.py : data loader with tf.python_io.TFrecord api

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

from ubicomp13 import parseTraces
import hyper_parameter



class parse_cambridge :
    def __init__(self):
        self.path_list_ = []
        self.label_list_ = []
        self.name_list_ = []
        self.channel_num = -1


    def get_name_from_path(self, l_n, path_list):
        item_num = len(path_list)
        name_l = []
        for j in range(0,item_num):
            toc = path_list[j].split('\\')[-1]
            name_l.append(toc)

        return name_l

    def label_extract(self,split):
        self.label_list_ = os.listdir('ubicomp13\\'+split+'_data\\')     # 폴더 목록을 list 로 반환
        class_num = len(self.label_list_)

        for i in range(0,class_num):

            #print('ubicomp13\\data\\'+label_list_[i]+'\\*.out')
            self.path_list_.append(glob('ubicomp13\\'+split+'_data\\'+self.label_list_[i]+'\\*.out'))
            self.name_list_.append(self.get_name_from_path(i,self.path_list_[i]))
        return self.name_list_, self.label_list_, self.path_list_

    def onehot_encode_label(self, label):
        onehot_label = []

        for iter_n in self.label_list_:
            # Boolean도 true false를 int로 변환하면 1, 0으로 변합니다.
            onehot_label.append(int(iter_n==label))

        return np.array(onehot_label).astype(np.uint8)


    def loading_data(self, path_list, height_ = 50, width_ = 9, num_channel=None, split='train'):
        '''
        :param name_list_:
        :param path_list_: 파일 경로 목록
        :param label_list_: 파일 레이블 목록
        :param split:
        :return:
        '''

        stacked_data = []
        stacked_label = []
        for path in path_list:
            item = len(path)
            nn = []
            data = []
            ll = []
            for it in range(item):
                label = os.path.dirname(path[it]).split('\\')[-1]

                (at, a, gt, g, mt, m) = parseTraces.parseTrace(path[it])

                n1 = len(at)
                l_at = len(at)
                l_a = len(a)
                l_g = len(g)
                l_m = len(m)
                # m_l = min([l_at,l_a,l_g,l_m])
                m_l = 6000
                # concatinate

                for k in range(m_l): # data 길이 k
                    nn.append(np.array([a[k][0], a[k][1], a[k][2], g[k][0], g[k][1], g[k][2], m[k][0], m[k][1], m[k][2]]))
                    if (k%height_==0):
                        onehotlabel = self.onehot_encode_label(label)
                        ll.append(onehotlabel)

            data = np.vstack(nn)
            all_len  = len(data)
            num_channel = int(all_len/(height_*width_))
            data = np.reshape(data, [-1, height_, width_])
            stacked_data.append(data)

            label_data = np.vstack(ll)
            stacked_label.append(label_data)

        stacked_data = np.vstack(stacked_data)
        stacked_label = np.vstack(stacked_label)

        return stacked_data, stacked_label






