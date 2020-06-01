# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:37:03 2020

@author: ASUS
"""

from IMDB_sentiment_Analysis import preprocess as pre, imdb, InitializeModel as im
import numpy as np
import matplotlib.pyplot as plt

'''
加载数据
num_words=1000:表示只保留训练数据中最常出现的10000个单词
train_data和test_data是评论列表，数据里的单词序列已被转换为整数序列，每个整数代表字典中的特定单词
train_labels和test_labels是0和1的列表，其中0表示‘负面评论’,1表示‘正面评论’
'''
'''
filename1 = 'Naive_Bayes_Meet_Adaboost/aclImdb/temp/neg'
filename = 'Naive_Bayes_Meet_Adaboost/aclImdb/temp/pos'
Words = pre.loadData(filename, filename1)
'''
print('reading vocabulary list')
word_index = imdb.get_word_index()
'''
Words = pre.setOfWordsListToVecTor(word_index, Words)
Words = pre.conf_data( x_train = Words, num_words=10000)
Words = pre.vectorize_sequences(Words)

'''
print('reading IMDB_data')
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# print('data0:', train_data[0], 'label0:', train_labels[0])

# print(max([max(sequence) for seq uence in train_data]))

# 将评论解码回到英文单词
# word_index是将单词映射到整数索引的字典

print(word_index)
# 将word_index反转，实现将整数索引到单词的映射

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print('decoded sample: ')
print(decoded_review)

'''
# 数据预处理
def vectorize_sequences(sequences, dimension=10000):
    # 创建一个全0矩阵->shape(len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
'''

# Vectoring data
print('Vectoring data')
X_train = pre.vectorize_sequences(train_data)
# X_test = pre.vectorize_sequences(test_data)
#
# # print(X_train[0])

# Vectorize label
print('Vectoring labels')
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# print(y_train[0])
'''
# Initialize model

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer=RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
'''
print('Initializing model')
model = im.modelConf()
X_val = X_train[: 10000]
partial_x_train = X_train[10000:]

y_val = y_train[: 10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(X_val, y_val))

history_dict = history.history
print(history_dict.keys())

# 绘制loss-acc图
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# figure show
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, acc, 'ro', label='Training acc')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Loss and Acc')
plt.xlabel('Epochs')
plt.ylabel('Loss and Acc')
plt.legend()

plt.show()
model.save('models/simple_model.h5')
