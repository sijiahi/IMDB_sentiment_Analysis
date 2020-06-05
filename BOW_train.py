import preprocess as pre
import imdb
import numpy as np
import matplotlib.pyplot as plt
import gc
import InitializeModel as im
import pickle

print('reading vocabulary list')
word_index = imdb.get_word_index()
# word_index = imdb.get_filtered_word_index()
print('reading IMDB_data')
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path='imdb', num_words=10000)
data = np.concatenate((train_data, test_data), axis=0)
targets = np.concatenate((train_labels, test_labels), axis=0)
'''
optional:Analyzing Dataset
'''
print("Categories:", np.unique(targets))
print("Number of unique words:", len(np.unique(np.hstack(data))))

# 将word_index反转，实现将整数索引到单词的映射

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print('decoded sample: ')
print(decoded_review)

# Simple Vectoring data
print('Vectoring data')
X_data = pre.vectorize_sequences(data, dimension=10000)
'''
# TF-IDF Vectoring data
print('Vectoring data')
X_train = tfidf.tf_idf(train_data)
X_test = tfidf.tf_idf(test_data)
'''
# Vectoring label
print('Vectoring labels')
y_label = np.asarray(targets).astype('float32')

test_x = X_data[: 10000]
train_x = X_data[10000:]

test_y = y_label[: 10000]
train_y = y_label[10000:]

model = im.default_modelConf(10000)
history = model.fit(train_x,
                    train_y,
                    epochs=10,
                    batch_size=512,
                    validation_data=(test_x, test_y))

evaluation = model.evaluate(test_x,
                            test_y,
                            batch_size=512)

history_dict = history.history
print(history_dict.keys())

# 绘制loss-acc图
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
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
model.save('models/BOW_default_40000.h5')
plt.savefig('report/BOW_default_40000.png')
with open('report/BOW_default_40000.txt', 'wb') as fhis:
    pickle.dump(history.history, fhis)
fhis.close()
gc.collect()