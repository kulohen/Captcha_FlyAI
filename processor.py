# -*- coding: utf-8 -*

from flyai.processor.base import Base
from path import *
import numpy as np
import cv2

'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''
    def input_x(self, img_path):
        img_path = os.path.join(DATA_PATH, img_path)
        img = cv2.imread(img_path)
        img = img.transpose(2, 0, 1)
        return img

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''
    def input_y(self, label):
        return label

    '''
    参数为csv中作为输入x的一条数据，该方c法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''
    def output_y(self, pred_label):
        return np.argmax(pred_label)