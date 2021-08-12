# 多个输入的神经网络
#################################

import numpy as np
import dataset
import plot_utils

m = 100
xs, ys = dataset.get_beans(m)

print(xs)
print(ys)
plot_utils.show_scatter(xs, ys)

w1 = 0.1
w2 = 0.1
b = 0.1

xs1 = xs[:0]
xs2 = xs[:1]


def forward_propagation(xs1, xs2):
    z = w1*xs1+w2*xs2+b
    a = 1/(1+np.exp(-z))
    return a

plot_utils.show_scatter_surface(xs, ys, forward_propagation)

for _ in range(500):
    for i in range(m):
        x = xs[i]
        y = ys[i]
        x1 = x[0]
        x2 = x[1]

        a = forward_propagation(x1, x2)

        e = (y-a)**2

        deda = -2*(y-a)
        dadz = a*(1-a)
        dzdw1 = x1
        dzdw2 = x2
        dzdb = 1

        dedw1 = deda*dadz*dzdw1
        dedw2 = deda*dadz*dzdw2
        dedb = deda*dadz*dzdb

        alpha = 0.1
        w1 = w1 - alpha*dedw1
        w2 = w2 - alpha*dedw2
        b = b - alpha*dedb

plot_utils.show_scatter_surface(xs, ys, forward_propagation)

