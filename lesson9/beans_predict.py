# 从数据中获取随机豆豆
from keras import Sequential
from keras.layers import Dense
from keras import optimizers
import tensorflow as tf

from lesson9 import dataset, plot_utils

m = 100
X, Y = dataset.get_beans(m)
print(X)
print(Y)

plot_utils.show_scatter(X, Y)

model = Sequential()

model.add(Dense(units=8, activation='relu', input_dim=2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics='accuracy')

model.fit(X, Y, epochs=5000, batch_size=10)

pres = model.predict(X)
plot_utils.show_scatter_surface(X, Y, model)
