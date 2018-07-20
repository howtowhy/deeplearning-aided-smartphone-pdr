

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
        self.channel_num = hyper_parameter.basic_parameters['channel']
        self.sample_num = hyper_parameter.basic_parameters['sample']


    def get_name_from_path(self,l_n,path_list):
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

    def onehot_encode_label(self,label):
        onehot_label = []
        iter = len(self.label_list_)
        for i in range(iter):
            if self.label_list_[i] == label:
                onehot_label.append(1)
            else:
                onehot_label.append(0)

        onehot_label = np.asarray(onehot_label)
        onehot_label = onehot_label.astype(np.uint8)
        return onehot_label


    def writing_data_to_tfrecord(self, name_list_, path_list_, label_list_, split): #파일 이름과 label list 를 받고, 상위 path 정보 받아서 path 생성
        class_num = len(label_list_)
        save_dir = "D:\\dev\\jejucamp-seoyeon\\classification\\ubicomp13_p"
        filename = os.path.join(save_dir, split + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(0, class_num): # 자세 class
            item_num = len(name_list_[i])
            for j in range(0,item_num):  # dataset 수 (안의 사람데이터)
                # make one hot label
                label_now = label_list_[i]
                onehotlabel = self.onehot_encode_label(label_now)

                # make data concatinate
                inname = name_list_[i][j]  # 이거를 네임 대신 list 로 받아야된다.
                inpath = path_list_[i][j]
                path = inpath
                (at,a,gt,g,mt,m) = parseTraces.parseTrace(path) # a1,
                n1 = len(at)
                # (accTs, accData, gyroTs, gyroData, magnTs, magnData)

                l_at = len(at)
                l_a = len(a)
                l_g = len(g)
                l_m = len(m)
               # m_l = min([l_at,l_a,l_g,l_m])
                m_l = 6000
                #concatinate
                nn = []
                for k in range(m_l): # data 길이 k
                    nn.append([at[k], a[k][0], a[k][1], a[k][2], g[k][0], g[k][1], g[k][2], m[k][0], m[k][1], m[k][2]])
                data = np.array(nn)
                data = np.reshape(data,[1000,10,6]) #batch, width, height, channel
                #print(data.shape)
                image = data.tostring()
                onehotlabel = np.array(onehotlabel)
                byte_onehot_label = onehotlabel.tostring()
                #inname = np.array(inname)
                # byte_inname = inname.tostring()
               # byte_inname = tf.compat.as_bytes(inname)
                chann = self.channel_num;
                sam = self.sample_num;

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                                    #'length': self._int64_feature(n1),
                                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_onehot_label])),
                                    'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[chann])),
                                    'sample': tf.train.Feature(int64_list=tf.train.Int64List(value=[sam])),
                                   # 'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_inname])),
                                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))}))
                writer.write(example.SerializeToString())

        writer.close()

        #parsing 해서 어떤 형태로 저장해서 tf.record 로 바꿀 것인가
        #멘토님께서 concate 하여서 하는 방법 제시한 것으로 구성해서 numpy type 으로 저장한다.

        return

    def read_data_from_tfrecode(self,name_):
        # READ
        NUM_EPOCHS = 10  # 중복?

        filename = os.path.join("D:\\dev\\jejucamp-seoyeon\\classification", name_+".tfrecords")

        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=NUM_EPOCHS)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.string),
                'channel' : tf.FixedLenFeature([], tf.int64),
                'sample': tf.FixedLenFeature([], tf.int64),
            #    'name': tf.FixedLenFeature([], tf.string),
                'image_raw': tf.FixedLenFeature([], tf.string),
            })


        image_ = tf.decode_raw(features['image_raw'], tf.float32) # 원래 타입 확인
        #image = tf.cast(image, tf.float32)
        image_ = tf.reshape(image_,[-1, 1000, 10, 6]) # 이건뭔가
        label_ = tf.decode_raw(features['label'], tf.uint8) # 타입확인
        label_ = tf.reshape(label_,[-1, 1, 6])  # 이건뭔가
        channel_ = tf.cast(features['channel'], dtype=tf.int32)
        sample_ = tf.cast(features['sample'], dtype=tf.int32)
        # name = tf.cast(features['name'], tf.string)
       # name = features['name'].value[0].decode('utf-8')#.bytes_list.value[0].decode('utf-8')
       # print(name)
 # def tfrecord_make(self):

        return image_, label_, channel_, sample_

    def preprocessing(self, image_p, label_p, channel_p, sample_p):



        return

    '''  dataset = tf.data.TFRecordDataset(image_)
      dataset = dataset.map(self._resize_function)
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=(int(len(image_) * 0.4) + 3 * 5))
      dataset = dataset.batch(5) # 5 batch size

      iterator = dataset.make_initializable_iterator()
      image_stacked, label_stacked = iterator.get_next() '''
