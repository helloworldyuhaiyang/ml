import dataset
import matplotlib.pyplot as plt

# 生成 x, y 的值数组
num = 100
xs, ys = dataset.get_beans(num)

# 图名
plt.title("size-toxicity")
# x 轴名
plt.xlabel("size")
# y 轴名
plt.ylabel("toxicity")
# 输入散点
plt.scatter(xs, ys)

# 函数为 y = wx , 求最合适的 w

w = 0.5  # 初始 w
alpha = 0.05
for m in range(100):
    for i in range(num):
        x = xs[i]
        y = ys[i]
        # 预测值
        y_pre = w * x
        # 误差值
        e = y - y_pre
        # 调整后的 w
        w = w + alpha * e * x

# 最终的 w 函数结果
y_pre = w * xs
plt.plot(xs, y_pre)
plt.show()
