from mpl_toolkits.mplot3d import Axes3D

from common import dataset
import matplotlib.pyplot as plt
import numpy as np

# 豆豆个数
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

# 预测函数
w = 0.1
b = 0.1
y_pre = w * xs + b
plt.plot(xs, y_pre)
plt.show()

# 3d 图
fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlim(0, 2)
# w 取不同的值
ws = np.arange(-1, 2, 0.1)
# b 取不同的值
bs = np.arange(-1, 2, 0.01)

for b in bs:
    # 均方误差
    es = []
    for w in ws:
        y_pre = w * xs + b
        e = np.sum((ys-y_pre) ** 2) * (1 / num)
        es.append(e)

    # 误差和 w 形成的函数
    ax.plot(ws, es, b, zdir='y')

plt.show()
