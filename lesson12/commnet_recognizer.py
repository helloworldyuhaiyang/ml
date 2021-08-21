import keras.optimizers

import shopping_data
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Dense, Embedding
from keras.losses import binary_crossentropy
from keras.optimizers import adam_v2

x_train, y_train, x_test, y_test = shopping_data.load_data()

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

print(x_train[0])
print(y_train[0])

# 所有词分割 生成索引表
vocalen, word_index = shopping_data.createWordIndex(x_train, x_test)

print("word_index:", word_index)
print("vocalen:", vocalen)

# 把文件句子转换为 词向量组成的 张量
x_train_index = shopping_data.word2Index(x_train, word_index)
x_test_index = shopping_data.word2Index(x_test, word_index)

# 填充成统一张量
maxlen = 25
x_train_index = sequence.pad_sequences(x_train_index, maxlen)
x_test_index = sequence.pad_sequences(x_test_index, maxlen)

# 送进神经网络中学习
# 初始化训练网络
model = Sequential()
model.add(Embedding(trainable=False, input_dim=vocalen, output_dim=300, input_length=maxlen))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))


model.add(Dense(units=1, activation='sigmoid'))
# 二分类交叉熵代价函数 适合二分类
loss = binary_crossentropy,
optimizer = keras.optimizers.adam_v2.Adam()
model.compile(loss=loss, optimizer=optimizer, metrics='accuracy')

# 开始训练
model.fit(x_train_index, y_train, epochs=100, batch_size=10)

# 使用测试集评估
loss_val, accuracy_val = model.evaluate(x_test_index, y_test)

print("loss_val:" + str(loss_val))
print("accuracy_val:" + str(accuracy_val))


