import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward
from testCases import *

np.random.seed(1)

#多层神经网络——初始化网络参数
def init_layer_parameters_deep(layers_dims):    #layers_dims包含网络中每一层中的节点数量
    #制作随机数种子
    np.random.seed(3)
    #定义存放参数的字典
    parameters = {}
    #层数
    L = len(layers_dims)-1
    for i in range(1,L+1):
        #W权重的初始化
        parameters['W' + str(i)] = np.random.randn(layers_dims[i],layers_dims[i-1])*0.01
        #b偏置的初始化
        parameters['b' + str(i)] = np.zeros((layers_dims[i],1))
        #断言
        assert (parameters['W' + str(i)].shape == (layers_dims[i],layers_dims[i-1]))
        assert (parameters['b' + str(i)].shape == (layers_dims[i],1))
    return parameters

#单层网络的前向传播
def liner_forward(A_pre,W,b,activation):
    #线性计算
    Z = np.dot(W, A_pre) + b
    #激活函数的前向传播
    if activation == 'sigmoid':
        A,activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A,activation_cache = relu(Z)
    #断言
    assert A.shape == (W.shape[0],A_pre.shape[1])
    #整合整个层的前向传播
    cache = (A_pre,W,b,activation_cache)
    return A,cache

#多层神经网络——前向传播函数
def L_forward_propagate_deep(X,params):
    A = X
    #缓存列表
    caches = []
    #层数
    L = len(params)//2
    #隐藏层的前向传播
    for l in range(1,L):
        #单层网络的前向传播
        A,cache = liner_forward(A,params['W'+str(l)],params['b'+str(l)],'relu')
        caches.append(cache)
    #输出层的前向传播
    output_A,cache = liner_forward(A,params['W'+str(L)],params['b'+str(L)],'sigmoid')
    caches.append(cache)
    #断言
    assert (output_A.shape == (params['b'+str(L)].shape[0],X.shape[1]))
    return output_A,caches

#多层神经网络——成本函数
def cost_deep(Y_hat,Y):     #单个样本的误差loss = -[y*log(a)+(1-y)*log(1-a)]
    #样本数
    m = Y.shape[1]
    #误差
    cost = -1/m * np.sum(Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat))
    cost = np.squeeze(cost)
    #断言
    assert (cost.shape == ())
    return cost

#单层网络的反向传播
def liner_backward(dA,cache,activation):
    #从内存中获取缓存的数据
    A_pre,W,b,activation_cache = cache
    #样本数
    m = A_pre.shape[1]
    #激活函数的反向传播
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
    #线性函数的反向传播
    dW = np.dot(dZ,A_pre.T) / m
    db = np.sum(dZ,axis=1,keepdims=True) / m
    dA_pre = np.dot(W.T,dZ) / m
    #断言
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    assert (dA_pre.shape == A_pre.shape)
    return dA_pre,dW,db

#对于成本函数为Loss = -[Y*log(A)+(1-Y)*log(1-A)],
#输出层的dAL = -[Y/AL-(1-Y)/(1-AL)]也可以写成：dAL = -(np.divide(Y,AL)-np.divide((1-Y),(1-AL)))
#多层神经网络——反向传播
def L_backward_deep(AL,Y,caches):
    #参数梯度
    grads = {}
    #层数
    L = len(caches)
    #重新规划Y的维度
    Y = Y.reshape(AL.shape)
    #输出层预测概率的梯度——输出层的输出值梯度
    dAL = -(np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))
    #输出层的参数梯度：grads[dA+str(L)]是输出层输入值的梯度
    grads['dA'+str(L)],grads['dW'+str(L)],grads['db'+str(L)] = liner_backward(dAL,caches[L-1],'sigmoid')
    #隐藏层的参数梯度
    for l in range(L-2,-1,-1):
        dA_prev,dW,db = liner_backward(grads['dA' + str(l+2)], caches[l], 'relu')
        grads['dA' + str(l+1)] = dA_prev
        grads['dW' + str(l+1)] = dW
        grads['db' + str(l+1)] = db
    return grads

#更新参数
def update_parameters(params,grads,learning_rate):
    #层数
    L = len(params)//2
    for l in range(1,L+1):
        params['W'+str(l)] = params['W'+str(l)] - learning_rate * grads['dW'+str(l)]
        params['b'+str(l)] = params['b'+str(l)] - learning_rate * grads['db'+str(l)]
    return params

#多层神经网络模型
def L_model_deep(X,Y,layers_dims,learning_rate=0.01,nums_iterations=3000,print_cost=False,isPlot=True):
    np.random.seed(1)
    #保存每次训练的误差
    costs = []
    #初始化参数
    parameters = init_layer_parameters_deep(layers_dims)
    #多次迭代
    for i in range(nums_iterations):
        #前向传播
        Y_hat,caches = L_forward_propagate_deep(X,parameters)
        #计算成本
        cost = cost_deep(Y_hat,Y)

        #反向传播
        grads = L_backward_deep(Y_hat,Y,caches)
        #参数更新
        parameters = update_parameters(parameters,grads,learning_rate)

        if i % 200 == 0:
            costs.append(cost)
            if print_cost:
                print('第%d次迭代，成本值为：%f'%(i,np.squeeze(cost)))
    #输出成本变化图
    if isPlot:
        plt.plot(costs)
        plt.show()
    return parameters

#评估函数
def evaluate(X,y,params):
    # 样本数
    m = X.shape[1]
    #预测结果
    predict_y = predict(X,params)
    print('准确度为：%f%%'%((1-np.sum(np.abs(y-predict_y))/m)*100))

#预测函数
def predict(X,params):
    # 样本数
    m = X.shape[1]
    # 预测结果保存的列表
    predict_y = np.zeros((1, m))
    # 预测
    y_hat, caches = L_forward_propagate_deep(X, params)
    for i in range(y_hat.shape[1]):
        if y_hat[0, i] > 0.5:
            predict_y[0, i] = 1
        else:
            predict_y[0, i] = 0
    return predict_y


if __name__ == '__main__':
    #测试L_model_deep
    print("==============测试L_model_deep==============")
    from lr_utils import load_dataset

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    train_y = train_set_y
    test_x = test_x_flatten / 255
    test_y = test_set_y

    parameters = L_model_deep(train_x,train_y,layers_dims=[12288,20,1],learning_rate=0.1,nums_iterations=3000,print_cost=True,isPlot=False)

    #评估
    predictions_train = evaluate(train_x,train_y,parameters)
    predictions_test = evaluate(test_x,test_y,parameters)






