# -*- coding: utf-8 -*
import numpy
import torch
from flyai.model.base import Base
from torch.autograd import Variable
from path import *
from net import se_resnet50

Torch_MODEL_NAME = "model.pkl"


cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(Base):
    def __init__(self, data):
        self.data = data

    def predict(self, **data):
        cnn = se_resnet50(num_classes=80)
        cnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME)))
        if cuda_avail:
            cnn.cuda()
        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data)
        x_data = x_data.float()
        if cuda_avail:
            x_data = Variable(x_data.cuda())
        outputs = cnn(x_data)
        outputs = outputs.cpu()
        prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        print(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
        cnn = se_resnet50(num_classes=80)
        cnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME)))
        cnn.to(device)
        cnn.eval()
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            x_data = torch.from_numpy(x_data)
            x_data = x_data.float().to(device)
            outputs = cnn(x_data)
            outputs = outputs.cpu()
            prediction = outputs.data.numpy()
            prediction = self.data.to_categorys(prediction)
            labels.append(prediction)
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=Torch_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network.state_dict(), os.path.join(path, name))