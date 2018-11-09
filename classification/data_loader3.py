

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
import math
from os import getcwd
import numpy as np
from six.moves import urllib
from glob import glob
from PIL import Image

from ubicomp13 import parseTraces
import hyper_parameter
from matplotlib import pyplot as plt
from scipy.fftpack import fft, rfft, irfft
from scipy import signal


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


    def loading_data(self, path_list, div_ =6, split='train', fs_=100, winsize_=20):
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

            data = []
            label = []
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
                a_x =[]
                a_y =[]
                a_z =[]
                g_x =[]
                g_y =[]
                g_z =[]
                m_x = []
                m_y = []
                m_z = []

                for k in range(m_l):
                    a_x.append(a[k][0])
                    a_y.append(a[k][1])
                    a_z.append(a[k][2])
                    g_x.append(g[k][0])
                    g_y.append(g[k][1])
                    g_z.append(g[k][2])
                    m_x.append(m[k][0])
                    m_y.append(m[k][1])
                    m_z.append(m[k][2])

                '''plt.plot(at[:6000],a_x)
                plt.title('Accel x Magnitude')
                plt.ylabel('Accel [m/s^2]')
                plt.xlabel('Time [sec]')
                plt.show()'''

                fs = fs_
                winsize = winsize_
                f, t, a_x_f = signal.stft(a_x, fs, nperseg=winsize)
                f, t, a_y_f = signal.stft(a_y, fs, nperseg=winsize)
                f, t, a_z_f = signal.stft(a_z, fs, nperseg=winsize)
                f, t, g_x_f = signal.stft(g_x, fs, nperseg=winsize)
                f, t, g_y_f = signal.stft(g_y, fs, nperseg=winsize)
                f, t, g_z_f = signal.stft(g_z, fs, nperseg=winsize)
                f, t, m_x_f = signal.stft(m_x, fs, nperseg=winsize)
                f, t, m_y_f = signal.stft(m_y, fs, nperseg=winsize)
                f, t, m_z_f = signal.stft(m_z, fs, nperseg=winsize)

                '''plt.pcolormesh(t, f, np.abs(a_x_f), vmin=0, vmax=3)
                plt.title('STFT Magnitude')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.show()'''

                concate_sen_f = [a_x_f,  a_y_f, a_z_f,  g_x_f,  g_y_f,  g_z_f,  m_x_f,  m_y_f,  m_z_f]
                concate_sen_f = np.vstack(concate_sen_f)
                 # data 길이 k
                m_l_a = concate_sen_f.shape[1]
                div = div_
                height_ = math.floor(m_l_a/div)

                for o in range(div):
                    nn = []
                    for k in range(height_):  # data 길이 k
                        kk = []
                        for j in range(concate_sen_f.shape[0]):
                            kk.append(concate_sen_f[j][o*height_+k])
                        nn.append(kk)
                    data.append(nn)
                    onehotlabel = self.onehot_encode_label(label)
                    ll.append(onehotlabel)
            data = np.asarray(data)
            label_data = np.vstack(ll)
            num_channel = len(data)
            #data = np.reshape(data, [height_, width_, -1])
            stacked_data.append(data)
            stacked_label.append(label_data)
        stacked_data = np.vstack(stacked_data)
        stacked_label = np.vstack(stacked_label)
        width_ = stacked_data.shape[2]
        return stacked_data, stacked_label, height_, width_






