# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:37:03 2020

@author: ASUS
"""

import keras
from IMDB_sentiment_Analysis import preprocess as pre, imdb
import numpy as np

print('Loading model')
model = keras.models.load_model('models/simple_model.h5')
print('Loading word_index')
word_index = imdb.get_word_index()
# word_index = pre.tag_words(word_index)

print('Loading test data')
filename1 = 'dataset/aclImdb/temp/neg'
filename = 'dataset/aclImdb/temp/pos'
Word, labels = pre.loadData(filename, filename1)
Words = pre.setOfWordsListToVecTor(word_index, Word)
Words = pre.conf_data(x_train=Words, num_words=10000)
Words = pre.vectorize_sequences(Words)
prediction = model.predict(Words)

prediction = np.ceil((prediction - 0.5))
prediction = prediction.astype(np.int)
prediction == labels
prediction = prediction.tolist()
ornot = [prediction[i][0] - labels[i] for i in range(len(labels))]
error = sum([abs(ornot[i]) for i in range(len(ornot))])
posError = 0
negError = 0
for ele in ornot:
    if ele < 0: posError += 1
    if ele > 0: negError += 1
print('Total error number:', posError + negError)
print('Total error rate:', (posError + negError) / len(prediction))
print('Positive error number:', posError)
print('Positive error rate:', posError / sum(1 for label in labels if label == 1))
print('Negative error number:', negError)
print('Negative error rate:', negError / sum(1 for label in labels if label == 0))
