import keras
from IMDB_sentiment_Analysis import preprocess as pre, imdb
import numpy as np
import matplotlib.pyplot as plt



print('Loading model')
model = keras.models.load_model('models/simple_model.h5')
print('reading vocabulary list')
word_index = imdb.get_word_index()
print('reading IMDB_data')
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(word_index)
# 将word_index反转，实现将整数索引到单词的映射

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print('decoded sample: ')
print(decoded_review)

# Vectoring data
print('Vectoring data')
X_train = pre.vectorize_sequences(train_data)
X_test = pre.vectorize_sequences(test_data)

# Vectoring label
print('Vectoring labels')
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
'''
X_val = X_test[: 10000]
partial_x_train = X_test[10000:]

y_val = y_test[: 10000]
partial_y_train = y_test[10000:]
'''

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128,
                    validation_data=(X_test, y_test))

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
plt.plot(epochs, val_loss, 'b', label='Valida tion loss')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Loss and Acc')
plt.xlabel('Epochs')
plt.ylabel('Loss and Acc')
plt.legend()

plt.show()
