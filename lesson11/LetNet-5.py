from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from keras import Sequential
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten

# 加载数据集
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("X_train shape:" + str(X_train.shape))
print("Y_train shape:" + str(Y_train.shape))
print("X_test shape:" + str(X_test.shape))
print("Y_test shape:" + str(Y_test.shape))

# 卷积神经网络必须使用图片才能卷
# 把数据集 样本的维度设置为 60000 * 28 * 28 *1(灰度的所以1通道)
X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0
# 把标签数据变为 one-host 数据
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# 初始化训练网络
model = Sequential()

# 增加卷积神经网络 变为 24*24*6
model.add(Conv2D(
    filters=6, kernel_size=(5, 5), strides=(1, 1),
    input_shape=(28, 28, 1), padding='valid', activation='relu'))
# 增加池化层 变为 12*12*6
model.add(AveragePooling2D(pool_size=(2, 2)))

# keras 是可以自动推倒此层的形状，所以 input_shape 不需要输入了
# 变为 8*8*6.
model.add(Conv2D(
    filters=6, kernel_size=(5, 5), strides=(1, 1),
    input_shape=(28, 28, 1), padding='valid', activation='relu'))

model.add(AveragePooling2D(pool_size=(2, 2)))

# 我们的全连接层需要输入一个数组。所以我们在最后的全链接层之前把数据平铺开(flatten)。
model.add(Flatten())

# 连接层神经网络
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 交叉熵代价函数 适合评估分类
loss = categorical_crossentropy,
optimizer = SGD(learning_rate=0.01)
model.compile(loss=loss, optimizer=optimizer, metrics='accuracy')

# 开始训练
model.fit(X_train, Y_train, epochs=100, batch_size=10)

# 使用测试集评估
loss_val, accuracy_val = model.evaluate(X_test, Y_test)

print("loss_val:" + str(loss_val))
print("accuracy_val:" + str(accuracy_val))

