

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


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class parse_cambridge :
    def __init__(self):
        self.path_list_ = []
        self.label_list_ = []
        self.name_list_ = []
        self.channel_num = hyper_parameter.basic_parameters['channel']
        self.sample_num = hyper_parameter.basic_parameters['sample']

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

    # 준호: 중요!!!!!! path_list는 from glob import glob 한 후에
    # path_list = glob('ubicomp_13/train_data/*/*.out')
    # 이런 식으로 만들어서 넣어주면 편합니다. glob가 이해가 안 되면 여쭤봐주세요!
    def writing_data_to_tfrecord(self, path_list, num_channel=6, split='train'): #파일 이름과 label list 를 받고, 상위 path 정보 받아서 path 생성
        '''
        :param name_list_:
        :param path_list_: 파일 경로 목록
        :param label_list_: 파일 레이블 목록
        :param split:
        :return:
        '''
        save_dir = "D:\\dev\\jejucamp-seoyeon\\classification\\ubicomp13_p"
        filename = os.path.join(save_dir, split + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)

        for path in path_list:
            item = len(path)
            for it in range(item):
                label = os.path.dirname(path[it]).split('\\')[-1]

                # 여기 부분은 이해가 안 되서 이 함수는 pass
                (at, a, gt, g, mt, m) = parseTraces.parseTrace(path[it])

                # 좀 더 간략하게 수정 드렸습니다.
                onehotlabel = self.onehot_encode_label(label)

                n1 = len(at)
                l_at = len(at)
                l_a = len(a)
                l_g = len(g)
                l_m = len(m)
                # m_l = min([l_at,l_a,l_g,l_m])
                m_l = 6000
                # concatinate

                nn = []
                for k in range(m_l): # data 길이 k
                    nn.append([at[k], a[k][0], a[k][1], a[k][2], g[k][0], g[k][1], g[k][2], m[k][0], m[k][1], m[k][2]])
                data = np.array(nn)

                height = 1000
                width = 10
                data = np.reshape(data, [height, width, num_channel])

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            # 준호: 중요팁! parsing 할 때의 dtype을 알아야 하므로, 넣을 때 미리 정해주고 넣으면 편합니다.
                            # tfrecords를 만들 때랑 parse 할 때 dtype이 꼭 맞아야함!!!!
                            'image_raw': _bytes_feature(data.astype(np.float32).tostring()),
                            # 여기 label은 데이터가 크지 않으므로 그냥 int로 바로 저장해도 좋습니다.
                            'label': _bytes_feature(onehotlabel.astype(np.int64).tostring()), #_int64_feature(onehotlabel),
                            'height': _int64_feature(height),
                            'width': _int64_feature(width),
                            'channel': _int64_feature(num_channel)
                            #'length': self._int64_feature(n1),
                            # 'name': _bytes_feature(byte_inname),
                        }))
                writer.write(example.SerializeToString())

        writer.close()
        return
        #parsing 해서 어떤 형태로 저장해서 tf.record 로 바꿀 것인가
        #멘토님께서 concate 하여서 하는 방법 제시한 것으로 구성해서 numpy type 으로 저장한다.
        # 준호: writer에는 return 안 해도 됩니다. 굳이 하려면 writer의 path를 return 해줘도 좋아요


    # 준호: param에 is_training이 있으면 shuffle 유무를 정해줄 수 있습니다.
    def read_data_from_tfrecode(self, name_, batch_size, num_epochs=10, is_training=True):
        filename = os.path.join("D:\\dev\\jejucamp-seoyeon\\classification", name_+".tfrecords")

        dataset = tf.data.TFRecordDataset([filename])


        def parser(record):
            keys_to_features = {
                'image_raw': tf.FixedLenFeature([], tf.string, default_value=''),
                'label': tf.FixedLenFeature([], tf.string, default_value=''),
                'height': tf.FixedLenFeature([], tf.int64, default_value=0),
                'width': tf.FixedLenFeature([], tf.int64, default_value=0),
                'channel' : tf.FixedLenFeature([], tf.int64, default_value=6),
                'sample': tf.FixedLenFeature([], tf.int64, default_value=0)
            #    'name': tf.FixedLenFeature([], tf.string),
            }

            parsed = tf.parse_single_example(record, keys_to_features)

            height = parsed['height']
            width = parsed['width']
            channel_ = parsed['channel']

            image_ = tf.decode_raw(parsed['image_raw'], tf.float32) # 원래 타입 확인
            image_ = tf.cast(image_, dtype=tf.float16)  # 준호: 학습시에는 float16 또는 float32로 변환하도록 합니다. // 왜요?
            # 준호: 한번 parse 할 때 마다 하나의 데이터씩만 뽑아야 합니다.
            image_ = tf.reshape(image_, [1000, 10, 6])

            label_ = tf.decode_raw(parsed['label'], tf.int64)  # 원래 타입 확인
            label_ = tf.cast(label_, dtype=tf.int64)  # 준호: 학습시에는 float16 또는 float32로 변환하도록 합니다.
            label_ = tf.reshape(label_, [1, 6])


            sample_ = tf.cast(parsed['sample'], dtype=tf.int32)

            return image_, label_, channel_, sample_

        dataset = dataset.map(parser)
        dataset = dataset.repeat()
        if is_training:
            dataset = dataset.shuffle(buffer_size=(100))  # 준호: buffer size는 직접 정해주시는게 ㅎㅎ
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()

        # 준호: 후에 데이터를 만들어 줄 때 next_element = iterator.get_next()로 한번 만들어 주고
        # iterator는 sess.run(iterator.initializer, feed_dict)로 한번 돌려주고
        # 데이터를 하나씩 꺼내고 싶을 땐 image = sess.run(next_element) 으로 돌리면 됩니다.
        return iterator


    def preprocessing(self, image_p, label_p, channel_p, sample_p):

        '''  dataset = tf.data.TFRecordDataset(image_)
          dataset = dataset.map(self._resize_function)
          dataset = dataset.repeat()
          dataset = dataset.shuffle(buffer_size=(int(len(image_) * 0.4) + 3 * 5))
          dataset = dataset.batch(5) # 5 batch size

          iterator = dataset.make_initializable_iterator()
          image_stacked, label_stacked = iterator.get_next() '''

        return image_p, label_p


