# 使用矩阵的方式实现 上节课的 多输入(特征的)的神经网络
#################################

import numpy as np
import dataset
import plot_utils

m = 100
X, Y = dataset.get_beans(m)

print(X)
print(Y)
plot_utils.show_scatter(X, Y)

W = np.array([0.1, 0.1])
B = np.array([0.1])


def forward_propagation(X):
    Z = X.dot(W.T) + B
    A = 1 / (1 + np.exp(-Z))
    return A


plot_utils.show_scatter_surface(X, Y, forward_propagation)

for _ in range(500):
    for i in range(m):
        # Xi 是一个一行两列的矩阵
        Xi = X[i]
        Yi = Y[i]
        # 前向传播
        A = forward_propagation(Xi)

        # 方差函数
        E = (Yi-A)**2

        dEdA = -2*(Yi-A)
        dAdZ = A*(1-A)
        dZdW = Xi
        dZdB = 1

        dEdW = dEdA*dAdZ*dZdW
        dEdB = dEdA*dAdZ*dZdB

        alpha = 0.1
        W = W - alpha*dEdW
        B = B - alpha*dEdB

plot_utils.show_scatter_surface(X, Y, forward_propagation)

