from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("X_train shape:" + str(X_train.shape))
print("Y_train shape:" + str(Y_train.shape))
print("X_test shape:" + str(X_test.shape))
print("Y_test shape:" + str(Y_test.shape))

# 不能把图片直接输入到链接层神经网络的
# 把数据集 样本的维度降为 60000 * 784
X_train = X_train.reshape(60000, 784)/255.0
X_test = X_test.reshape(10000, 784)/255.0
# 把标签数据变为 one-host 数据
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

model = Sequential()

model.add(Dense(units=8, activation='relu', input_dim=784))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=10, activation='softmax'))


# 交叉熵代价函数 适合评估分类
loss = categorical_crossentropy,
sgd = SGD(learning_rate=0.01)
model.compile(loss=loss, optimizer=sgd, metrics='accuracy')

# 开始训练
model.fit(X_train, Y_train, epochs=100, batch_size=10)

# 使用测试集评估
loss_val, accuracy_val = model.evaluate(X_test, Y_test)

print("loss_val:" + str(loss_val))
print("accuracy_val:" + str(accuracy_val))
