# 从数据中获取随机豆豆
import tensorflow
from keras import Sequential
from keras.layers import Dense
from keras import optimizers

from lesson9 import dataset, plot_utils

m = 100
X, Y = dataset.get_beans(m)
print(X)
print(Y)

plot_utils.show_scatter(X, Y)

# 构建网络
model = Sequential()

model.add(Dense(units=8, activation='relu', input_dim=2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# 设置反向传播的参数
sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics='accuracy')

# 进行训练
model.fit(X, Y, epochs=5000, batch_size=10)

pres = model.predict(X)
plot_utils.show_scatter_surface(X, Y, model)
