import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#导入数据
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()
#将特征数据扁平化
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#将特征数据归一化
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
#Logisitic----sigmoid函数
def sigmoid(z):
    s = 1 / (1+np.exp(-z))
    return s

#定义传播函数
def propagation(W,b,X,Y):
    #样本数
    m = X.shape[1]
    #假设函数
    Z = np.dot(W.T,X) + b
    A = sigmoid(Z)
    #代价函数
    cost = (-1/m) * np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    #参数w的梯度
    dw = 1/m * np.dot(X,(A-Y).T)
    #参数b的梯度
    db = 1/m * np.sum(A-Y)
    assert(dw.shape == W.shape)
    cost = np.squeeze(cost)
    grads = {'dw':dw,
             'db':db}
    return grads,cost
#优化函数
def optimize(W,b,X,Y,num_iteration,learning_rate,print_cost = False):
    costs = []
    for i in range(num_iteration):
        grads,cost = propagation(W,b,X,Y)
        dw = grads['dw']
        db = grads['db']
        W = W - learning_rate*dw
        b = b - learning_rate*db
        ##
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print('cost after iteration %i:%f'%(i,cost))
    params = {'W':W,
              'b':b}
    grads = {'dw':dw,
             'db':db}
    return params,grads,costs
#预测函数
def predict(W,b,X):
    #样本数
    m = X.shape[1]
    #预测数组
    Y_prediction = np.zeros(shape=(1,m))
    #为了保证矩阵维度的正确，所以再次变换当需要的维度上()
    W = W.reshape(X.shape[0],1)
    #预测值
    A = sigmoid(np.dot(W.T,X)+b)
    for i in range(A.shape[1]):
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0
    assert(Y_prediction.shape == (1,m))
    return Y_prediction
#整个训练模型
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # 初始化参数矩阵
    W = np.zeros(shape=(X_train.shape[0], 1))
    b = 0
    #参数、参数梯度以及代价函数值
    params,grads,costs = optimize(W,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    #更新参数
    W = params['W']
    b = params['b']
    #预测
    Y_prediction_test = predict(W,b,X_test)
    Y_prediction_train = predict(W,b,X_train)
    #输出
    print('train accuracy: {}%'.format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print('test accuracy: {}%'.format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))

    d = {'costs':costs,
         'Y_prediction_test':Y_prediction_test,
         'Y_prediction_train':Y_prediction_train,
         'W':W,
         'b':b,
         'learning_rate':learning_rate,
         'num_iteration':num_iterations}
    return d
#显示的查看一个样本
def show_one_sample(index):
    plt.imshow(train_set_x_orig[index])
    plt.show()
#显示样本的shape
def show_samples_shape():
    print('train_set_x shape:' + str(train_set_x_orig.shape))
    print('train_set_y shape:' + str(train_set_y.shape))
    print('test_set_x shape:' + str(test_set_x_orig.shape))
    print('test_set_y shape:' + str(test_set_y.shape))

if __name__ == '__main__':
    #Test = model(train_set_x,train_set_y,test_set_x,test_set_y,learning_rate=0.01,print_cost=False)
    # index = 8
    # plt.imshow(test_set_x[:,index].reshape((64,64,3)))
    # print("y="+str(test_set_y[0,index])+" predicted is "+classes[int(Test['Y_prediction_test'][0,index])].decode('utf-8'))
    # plt.show()
    # costs = np.squeeze(Test['costs'])
    # plt.plot(costs)
    # plt.show()
    show_samples_shape()


