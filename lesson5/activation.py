# 加入了激活函数的 神经元
#################################

import numpy as np

import matplotlib.pyplot as plt

# 豆豆个数
from lesson5 import dataset

num = 100
# 生成 x, y 的值数组
xs, ys = dataset.get_beans(num)

# 图名
plt.title("size-toxicity")
# x 轴名
plt.xlabel("size")
# y 轴名
plt.ylabel("toxicity")
# 输入散点
plt.scatter(xs, ys)

# 复合函数
w = 0.1
b = 0.1
z = w * xs + b
# sigmoid(z)
a = 1 / (1 + np.exp(-z))
# 画线
plt.plot(xs, a)

plt.show()

for _ in range(500):
    # 调整参数过程
    alpha = 0.01  # 学习率
    for i in range(num):
        x = xs[i]
        y = ys[i]
        # w,b 导数
        z = w * x + b
        a = 1 / (1 + np.exp(-z))
        e = (y - a) ** 2
        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw = x

        dedw = deda * dadz * dzdw

        dzdb = 1
        dedb = deda * dadz * dzdb

        # 进行梯度下降
        w = w - alpha * dedw
        b = b - alpha * dedb

    # 重新绘制图像
    if _ % 100 == 0:
        plt.clf()
        plt.scatter(xs, ys)
        z = w * xs + b
        a = 1 / (1 + np.exp(-z))
        # 限制横纵坐标空间
        plt.xlim(0, 1)
        plt.ylim(0, 1.2)
        plt.plot(xs, a)
        plt.pause(0.01)
