# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from net import se_resnet50
from flyai.dataset import Dataset
from torch.optim import Adam
from model import Model
from path import *

# 导入flyai打印日志函数的库
from flyai.utils.log_helper import train_log



# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=24, type=int, help="batch size")
args = parser.parse_args()

# 数据获取辅助类
# dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
dataset = Dataset(epochs=100, batch=32)

# 模型操作辅助类
flyai_tool = Model(dataset)

# 判断gpu是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(model_net, flyai_tool, x_test, y_test):
    model_net.eval()
    batch_eval = flyai_tool.batch_iter(x_test, y_test)
    total_acc = 0.0
    data_len = len(x_test)
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        outputs = model_net(x_batch)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
    return total_acc / data_len

# 训练并评估模型
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

model_net = se_resnet50(num_classes=80)
model_net.to(device)

optimizer = Adam(model_net.parameters(), lr=0.001, betas=(0.9, 0.999))  # 选用AdamOptimizer
loss_fn = nn.CrossEntropyLoss().to(device)  # 定义损失函数

best_accuracy = 0
num_all_steps = dataset.get_step()

print(model_net)

for i in range(num_all_steps):

    model_net.train()
    x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)  # 读取数据

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float().to(device)
    y_train = y_train.long().to(device)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_test = x_test.float().to(device)
    y_test = y_test.long().to(device)

    # 打印步骤和训练集/测试集的量
    cur_step = str(i + 1) + "/" + str(num_all_steps)
    print('\n步骤' + cur_step, ': %d on train, %d on val' % (len(x_train), len(x_test)))

    outputs = model_net(x_train)
    _, prediction = torch.max(outputs.data, 1)

    optimizer.zero_grad()
    # print(x_train.shape,outputs.shape,y_train.shape)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()
    # print(loss)
    train_loss_list.append(loss.item())
    # print('loss is ',loss.item())

    train_loss = 0
    correct_total = 0
    max_acc = 0
    last_loss = 0


    predict = torch.argmax(outputs, 1)
    correct_total += torch.eq(predict, y_train).sum().item()
    train_loss += loss.item()

    train_acc = correct_total / len(x_train)
    val_acc = eval(model_net, flyai_tool, x_test, y_test)
    train_acc_list.append(train_acc)
    # 若测试准确率高于当前最高准确率，则保存模型
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        flyai_tool.save_model(model_net, MODEL_PATH, overwrite=True)
        print("step %d, all step %d, best accuracy %g" % (i, num_all_steps, best_accuracy))

    # 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
    # train_log(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
    train_log(train_loss=train_loss, train_acc=train_acc, val_acc=val_acc)
    # train_log(train_loss=loss.item(), train_acc=train_accuracy)