import keras
import preprocess as pre
import imdb
import numpy as np
import matplotlib.pyplot as plt
import gc
import InitializeModel as im
import tf_idf as tfidf
import scipy.sparse as sp
import pickle
from tf_idf import stemmed_words

'''
print('Loading model')
model = keras.models.load_model('models/simple_model.h5')
'''

print('reading IMDB_data')
(train_data, train_labels) = pre.loadData('dataset/aclImdb/train/pos', 'dataset/aclImdb/train/neg')
(test_data, test_labels) = pre.loadData('dataset/aclImdb/test/pos', 'dataset/aclImdb/test/neg')
'''
optional:Analyzing Dataset
'''
print("Categories:", np.unique(train_labels))
print("Number of unique words:", len(np.unique(np.hstack(train_data))))

# 将word_index反转，实现将整数索引到单词的映射
'''
# Simple Vectoring data
print('Vectoring data')
X_train = pre.vectorize_sequences(train_data)
X_test = pre.vectorize_sequences(test_data)
'''
# TF-IDF Vectoring data
print('\nVectoring train data')
X_train, train_labels = tfidf.tf_idf_2doc(train_data, train_labels, feat=10000)
print('\nVectoring test data')
X_test, test_labels = tfidf.tf_idf_2doc(test_data, test_labels, feat=10000)
data = sp.vstack((X_train, X_test))

# Vectoring label
print('\nVectoring labels')
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
target = np.append(y_train, y_test)
'''
X_val = X_test[: 10000]
partial_x_train = X_test[10000:]

y_val = y_test[: 10000]
partial_y_train = y_test[10000:]
'''
train_x = data[10000:]
train_y = target[10000:]
test_x = data[:10000]
test_y = target[:10000]

print('\nInitializing model')
model = im.default_modelConf(size=10000)
print('\nTraining model')
history = model.fit(train_x,
                    train_y,
                    epochs=11,
                    batch_size=500,
                    validation_data=(test_x, test_y))
history_dict = history.history
print(history_dict.keys())

# figure show
# "bo" is for "blue dot"
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['accuracy'], 'ro', label='Training acc')
# b is for "solid blue line"
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation acc')
plt.title('Loss and Acc')
plt.xlabel('Epochs')
plt.ylabel('Loss and Acc')
plt.legend()
plt.grid()
plt.show()
model.save('models/tfidf_default_40000.h5')
plt.savefig('report/tfidf_default_40000.png')
with open('report/tfidf_default_40000.txt', 'wb') as fhis:
    pickle.dump(history.history, fhis)
fhis.close()
gc.collect()
