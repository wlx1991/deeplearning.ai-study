import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model as slm
from testCases import *
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets


#导入数据
x,y = load_planar_dataset() #x为特征矩阵，y为标签
#绘制散点图
# plt.scatter(x[0,:],x[1,:],c=y)
# plt.show()

#Sigmoid激活函数
def sig(z):
    s = 1 / (1+np.exp(-z))
    return s

#预测函数
def prediction(W,b,X):
    #样本数量
    m = X.shape[1]
    #让W参数向量的维度保证是与X向量一致
    W = W.reshape((X.shape[0],1))
    #预测矩阵
    prediction_y = np.zeros((1,m))
    #单个样本的预测值
    A = sig(np.dot(W.T,X) + b)
    #所有预测
    for i in range(A.shape[1]):
        prediction_y[0][i] = 1 if A[0][i] > 0.5 else 0
    assert (prediction_y.shape == (1,m))
    return prediction_y

#传播函数
def propagation(W,b,X,Y):
    #样本数量
    m = X.shape[1]
    #假设函数
    A = sig(np.dot(W.T,X) + b)
    #代价函数
    cost = -(1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    #参数的偏导数
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)

    assert (dw.shape == W.shape)
    cost = np.squeeze(cost)
    grads = {'dw':dw,
            'db':db}
    return cost,grads

#优化函数——梯度下降法
def optimize_GD(W,b,X,Y,num_iteration,rating,print_cost = False):
    #误差列表
    costs = []
    for i in range(num_iteration):
        #调用传递函数
        cost,grads = propagation(W,b,X,Y)
        #参数导数
        dw = grads['dw']
        db = grads['db']
        #参数更新
        W = W - rating * dw
        b = b - rating * db
        if i % 100 == 0:
            costs.append(cost)
        #判读是否输出cost
        if print_cost and i % 100 == 0:
            print('cost after iteration %i:%f'%(i,cost))

    #参数字典
    params = {'W':W,
              'b':b}
    #参数梯度
    grads = {'dw':dw,
             'db':db}
    return params,grads,costs   #costs列表是为了可视化优化结果

def mode(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    #初始化参数
    W = np.random.randn(X_train.shape[0],1)
    b = 0
    #
    params,grads,costs = optimize_GD(W,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    #更新参数
    W = params['W']
    b = params['b']
    #预测
    prediction_y_test = prediction(W,b,X_test)
    prediction_y_train = prediction(W,b,X_train)
    #
    print('test accuracy: %.2f%%'%((1-np.mean(np.abs(prediction_y_test-Y_test)))*100))
    print('train accuracy: %.2f%%'%((1-np.mean(np.abs(prediction_y_train-Y_test)))*100))
    #模型输出
    clf = {'costs':costs,
           'W':W,
           'b':b,
           'prediction_y_test':prediction_y_test,
           'prediction_y_train':prediction_y_train,
           'num_iterations':num_iterations,
           'learning_rate':learning_rate}
    return clf

if __name__ == '__main__':
    d = mode(x,y,x,y,num_iterations=500,learning_rate=0.1,print_cost=False)
    plt.plot(d['costs'])
    plt.title('Logistic Costs')
    plt.show()

    # #加载查看数据集
    # x,y = load_planar_dataset()
    # #绘制散点图
    # # plt.scatter(x[0,:],x[1,:],c=y,s=40,cmap=plt.cm.Spectral)
    # # plt.show()
    # #先使用Logistic回归预测
    # clf_L = slm.LogisticRegressionCV()
    # clf_L.fit(x.T,y.T)
    # #预测
    # LR_predictions = clf_L.predict(x.T)
    # #预测类别为1的正确个数
    # true_num_1 = np.dot(y,LR_predictions)
    # #预测类别为0的正确个数
    # true_num_0 = np.dot(1-y,1-LR_predictions)
    # #预测正确的概率
    # accuracy_rate = (true_num_1+true_num_0) / y.shape[1] * 100
    # print('Logistic回归预测的准确率：%.2f%%' % accuracy_rate)