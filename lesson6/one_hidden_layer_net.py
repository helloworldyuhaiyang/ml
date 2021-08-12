# 单层隐藏神经网络
# 权重命名规范
# w12_3 在第三层第二个神经元第一个输入参数的权重
# b2_3 第三层第二个神经元 的偏置项

#################################


import numpy as np

import matplotlib.pyplot as plt

# 豆豆个数
from lesson6 import dataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

# 第一层
# 第一个神经元
w11_1 = np.random.rand()
b1_1 = np.random.rand()
# 第一层
# 第二个神经元
w12_1 = np.random.rand()
b2_1 = np.random.rand()
# 第二层
w11_2 = np.random.rand()
w21_2 = np.random.rand()
b1_2 = np.random.rand()


# 前线传播
def forward_propagation(xs):
    z1_1 = w11_1 * xs + b1_1
    a1_1 = sigmoid(z1_1)

    z2_1 = w12_1 * xs + b2_1
    a2_1 = sigmoid(z2_1)

    z1_2 = w11_2 * a1_1 + w21_2 * a2_1 + b1_2
    a1_2 = sigmoid(z1_2)
    return a1_2, z1_2, a2_1, z2_1, a1_1, z1_1


a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propagation(xs)

plt.plot(xs, a1_2)
plt.show()


alpha = 0.03  # 学习率

for _ in range(50000):
    # 调整参数过程
    for i in range(num):
        x = xs[i]
        y = ys[i]
        # 前向传播
        a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propagation(x)

        # 反向传播
        # 第三层  e = (y-a1_2)**2
        deda1_2 = -2 * (y - a1_2)
        da1_2dz1_2 = a1_2 * (1 - a1_2)
        dz1_2dw11_2 = a1_1
        dz1_2dw21_2 = a2_1
        dedw11_2 = deda1_2 * da1_2dz1_2 * dz1_2dw11_2
        dedw21_2 = deda1_2 * da1_2dz1_2 * dz1_2dw21_2
        dz1_2db1_2 = 1
        dedb1_2 = deda1_2 * da1_2dz1_2 * dz1_2db1_2

        # 第二层  第一个神经元
        dz1_2da1_1 = w11_2
        da1_1dz1_1 = a1_1 * (1 - a1_1)
        dz1_1dw11_1 = x
        dedw11_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da1_1dz1_1 * dz1_1dw11_1

        dz1_1db1_1 = 1
        dedb1_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da1_1dz1_1 * dz1_1db1_1

        # 第二层 第二个神经元
        dz1_2da2_1 = w21_2
        da2_1dz2_1 = a2_1 * (1 - a2_1)
        dz2_1dw12_1 = x
        dedw12_1 = deda1_2 * da1_2dz1_2 * dz1_2da2_1 * da2_1dz2_1 * dz2_1dw12_1

        dz2_1db2_1 = 1
        dedb2_1 = deda1_2 * da1_2dz1_2 * dz1_2da2_1 * da2_1dz2_1 * dz2_1db2_1

        w11_2 = w11_2 - alpha * dedw11_2
        w21_2 = w21_2 - alpha * dedw21_2
        b1_2 = b1_2 - alpha * dedb1_2

        w12_1 = w12_1 - alpha * dedw12_1
        b2_1 = b2_1 - alpha * dedb2_1

        w11_1 = w11_1 - alpha * dedw11_1
        b1_1 = b1_1 - alpha * dedb1_1

    # 重新绘制图像
    if _ % 200 == 0:
        plt.clf()
        plt.scatter(xs, ys)
        print(f'w11_2:{w11_2}, w21_2:{w21_2}, b1_2:{b1_2}, w12_1:{w12_1}, b2_1:{b2_1}, w11_1:{w11_1}, b1_1:{b1_1}')
        ypre, _, _, _, _, _ = forward_propagation(xs)
        # print(f'xs:{xs}')
        # print(f'ypre:{ypre}')
        plt.plot(xs, ypre)
        plt.pause(0.01)
