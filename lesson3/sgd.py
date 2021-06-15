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

# plt.show()

# 调整参数过程
for i in range(num):
    x = xs[i]
    y = ys[i]
    # 代价函数的各项 a=x^2, b=-2xy, c=y^2
    # 斜率 k=2aw+b
    k = 2*(x**2)*w + (-2*x*y)
    # 学习率
    alpha = 0.1
    # 进行地图下降
    w = w-alpha*k
    # 重新绘制图像
    plt.clf()
    plt.scatter(xs, ys)
    y_pre = w*xs
    # 限制横纵坐标空间
    plt.xlim(0, 1)
    plt.ylim(0, 1.2)
    plt.plot(xs, y_pre)
    plt.pause(0.01)
