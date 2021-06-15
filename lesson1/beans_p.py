from common import dataset
import matplotlib.pyplot as plt

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
# 假设函数 y = 0.5*x
w = 0.5
y_pre = w * xs
# 画线
plt.plot(xs, y_pre)

plt.show()