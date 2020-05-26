#-*- coding: utf-8 -*-
#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset
from neuralNetworkM import sigmoid
from neuralNetworkM import propagate
from neuralNetworkM import initialize_with_zeros
from neuralNetworkM import optimize
from neuralNetworkM import predict
from neuralNetworkM import model

import pylab


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# index = 25
# plt.imshow(train_set_x_orig[index])
# # 显示图片
# #pylab.show()
# print("y=" + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture")
# m_train = train_set_y.shape[1] #训练集里图片的数量。
# m_test = test_set_y.shape[1] #测试集里图片的数量。
# num_px = train_set_x_orig.shape[1] #训练、测试集里面的图片的宽度和高度（均为64x64）。
#
# #现在看一看我们加载的东西的具体情况
# print ("训练集的数量: m_train = " + str(m_train))
# print ("测试集的数量 : m_test = " + str(m_test))
# print ("每张图片的宽/高 : num_px = " + str(num_px))
# print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
# print ("训练集_标签的维数 : " + str(train_set_y.shape))
# print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
# print ("测试集_标签的维数: " + str(test_set_y.shape))

# 这一段意思是指把数组变为209行的矩阵（因为训练集里有209张图片），但是我懒得算列有多少，吴恩达课程中找特征向量的方法
# 于是我就用-1告诉程序你帮我算，最后程序算出来时12288列，我再最后用一个T表示转置，这就变成了12288行，209列。
# 测试集亦如此。
#X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
#将训练集的维度降低并转置。
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T #train_set_x_orig.shape[0] 209张图片代表有209行
#将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
# print ("训练集_标签的维数 : " + str(train_set_y.shape))
# print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
# print ("测试集_标签的维数 : " + str(test_set_y.shape))

#因为在RGB中不存在比255大的数据，所以我们可以放心的除以255，让标准化的数据位于[0,1]之间，现在标准化我们的数据集
train_set_x = train_set_x_flatten/255.  #这里一定要是255.不能是255,整了一下午终于找到了
test_set_x = test_set_x_flatten/255.
# print("====================测试sigmoid====================")
# print ("sigmoid(0) = " + str(sigmoid(0)))
# print ("sigmoid(9.2) = " + str(sigmoid(9.2)))
#
# #测试一下propagate
# print("====================测试propagate====================")
#
# #初始化一些参数
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# grads, cost = propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))
#
#
# print("====================测试optimize====================")
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = False)
# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
#
# #测试predict
# print("====================测试predict====================")
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# print("predictions = " + str(predict(w, b, X)))
#
# print("====================测试model====================")


# 单独一种学习速率开始训练模型
# d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
# #绘制图
# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()

#比较多种学习速率曲线的变化
learning_rates = [0.03, 0.01, 0.001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

#可以看出学习速率在0.01左右的时候的时候是最好的，