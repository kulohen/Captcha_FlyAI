# -*- coding: utf-8 -*
import sys
import os

class Config(object):
    # 训练数据的路径
    DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
    # 模型保存的路径
    MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
    # 训练log的输出路径
    LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')
    # 保存模型名字
    Torch_MODEL_NAME = "model.pkl"




config = Config()