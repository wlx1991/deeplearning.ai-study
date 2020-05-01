from  planar_utils import plot_decision_boundary,load_planar_dataset,load_extra_datasets,sigmoid
from testCases import *
import numpy as np
import matplotlib.pyplot as plt

#定义神经网络的结构
def layer_sizes(X,Y):   #X为输入，Y为输出
    #输入层的节点个数
    input_layer_sizes = X.shape[0]
    #隐藏层的节点个数
    hidden_layer_sizes = 4
    #输出层的节点个数
    output_layer_sizes = Y.shape[0]
    return input_layer_sizes,hidden_layer_sizes,output_layer_sizes

#初始化模型参数
def init_params(n_x,n_h,n_y):   #n_x为输出层的节点个数，n_h为隐藏层的节点个数，n_y为输出层的节点个数
    #随机数种子
    np.random.seed(2)
    #初始化隐藏层中的参数
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros(shape=(n_h,1))
    #初始化输出层中的参数
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(shape=(n_y,1))
    #断言,判断参数维数设定是否正确
    assert (W1.shape == (n_h,n_x))
    assert (b1.shape == (n_h,1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    params = {'W1':W1,
              'b1':b1,
              'W2':W2,
              'b2':b2}
    return params

#前向传播函数
def forward_propagation(X,params):
    #隐藏层的参数
    W1 = params['W1']
    b1 = params['b1']
    #输出层的参数
    W2 = params['W2']
    b2 = params['b2']
    #隐藏层的前向传播计算
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)    #使用tanh()为隐藏层的激活函数
    #输出层的前向传播计算
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)    #使用sigmoid()为输出层的激活函数

    #断言判读输出维度是否正确
    assert (A2.shape == (1,X.shape[1]))
    cache = {'Z1':Z1,
             'A1':A1,
             'Z2':Z2,
             'A2':A2}
    return cache

#代价函数
def cost_J(A2,Y,params):
    #样本数量
    m = Y.shape[1]
    #单个样本的成本函数
    log_cost = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    #整体样本的代价函数
    cost = -1/m * np.sum(log_cost)
    cost = float(np.squeeze(cost))
    #断言
    assert (isinstance(cost,float))
    return cost

#反向传播函数
def backward_propagation(X,Y,cache,params):
    #样本数
    m = X.shape[1]
    #隐藏层和输出层中W
    W1 = params['W1']
    W2 = params['W2']
    #前向传播的输出结果
    A1 = cache['A1']
    A2 = cache['A2']
    #输出层的反向计数
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2,axis=1,keepdims=True)
    #隐藏层的反向计算
    dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
    dW1 = 1/m * np.dot(dZ1,X.T)
    db1 = 1/m * np.sum(dZ1,axis=1,keepdims=True)
    #断言
    assert (dW1.shape == W1.shape)
    assert (dW2.shape == W2.shape)

    grads = {'dW1':dW1,
             'dW2':dW2,
             'db1':db1,
             'db2':db2}
    return grads

#参数更新函数——梯度下降法
def update_params_gb(params,grads,learning_rate):
    #前向传播的参数
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
    #反向传播的参数
    dW1 = grads['dW1']
    dW2 = grads['dW2']
    db1 = grads['db1']
    db2 = grads['db2']
    #更新前向传播参数
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    #返回新的参数
    params = {'W1':W1,
              'b1':b1,
              'W2':W2,
              'b2':b2}
    return params

#单隐藏层的神经网络模型
def small_NN_model(X,Y,n_h,num_iterations,learning_rate=0.5,print_cost=False):
    np.random.seed(3)
    #输入，输出节点个数
    n_x,_,n_y = layer_sizes(X,Y)
    #初始化前向传播的参数
    params = init_params(n_x,n_h,n_y)
    #迭代
    for i in range(num_iterations):
        #前向传播
        cache = forward_propagation(X,params)

        #代价函数
        cost = cost_J(cache['A2'],Y,params)

        #反向传播
        grads = backward_propagation(X,Y,cache,params)
        #更新参数
        params = update_params_gb(params,grads,learning_rate)
        if print_cost and i%1000==0:
            print('第 %d 次循环，代价值：%f'%(i,cost))
    return params

#预测函数
def predict(X,params):
    #前向传播
    cache = forward_propagation(X,params)
    predict_Y = cache['A2']

    # for i in range(predict_Y.shape[1]):
    #     if predict_Y[0,i] > 0.5:
    #         predict_Y[0,i] = 1
    #     else:
    #         predict_Y[0,i] = 0
    predict_Y = np.round(predict_Y)
    return predict_Y



if __name__ == '__main__':
    #测试一下init_params()
    # n_x,n_h,n_y = initialize_parameters_test_case()
    # params = init_params(n_x,n_h,n_y)
    # print(params['W1'])
    # print(params['b1'])
    # print(params['W2'])
    # print(params['b2'])
    #测试一下反向传播函数
    # params,cache,X_assess,Y_assess = backward_propagation_test_case()
    # grads = backward_propagation(X_assess,Y_assess,cache,params)
    # print(grads['dW1'])
    # print(grads['dW2'])
    # print(grads['db1'])
    # print(grads['db2'])
    # 测试nn_model
    print("=========================测试nn_model=========================")
    X,Y = load_planar_dataset()
    print('样本个数：%d'%X.shape[1])
    #随机获取训练样本和测试样本
    index_1 = np.random.choice(X.shape[1],280,replace=False)    #第一个参数是好像必须是一维数组或是Int变量,所以随机获取的是原样本的下标，
                                                                # replace=False时，就不会获取重复的数据
    index_2 = np.delete(np.array([i for i in range(X.shape[1])]),index_1)   #从原有的一维下标数组中删除以获取的下标
    x_train = X[:,index_1]
    y_train = Y[:,index_1]
    x_test = X[:,index_2]
    y_test = Y[:,index_2]


    parameters = small_NN_model(x_train, y_train, 50, num_iterations=10000, learning_rate=0.6,print_cost=True)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    #测试predict()
    predictions = predict(x_test,parameters)
    print('准确度：%.2f%%'%((1-np.mean(np.abs(y_test-predictions)))*100))
    # #print(predictions)
