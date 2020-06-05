# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:37:03 2020

@author: ASUS
"""
import re

import keras
import imdb
import numpy as np
import preprocess as pre
import tkinter

# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:37:03 2020

@author: ASUS
"""

import keras
import imdb
import numpy as np
import preprocess as pre
import tf_idf as tfidf
import pickle
import matplotlib.pyplot as plt
import requests
from requests.exceptions import RequestException
import tkinter as tk


def predict(modelType='tfidf', feat=10000, Words=None, labels=None):
    if modelType == 'BOW':
        print('Loading model')
        model = keras.models.load_model('models/BOW_default_40000.h5')
        word_index = imdb.get_word_index()
        print('\nPreprocessing data')
        Words = pre.setOfWordsListToVecTor(word_index, Words)
        Words = pre.conf_data(x_train=Words, num_words=10000)
        Words = pre.vectorize_sequences(Words)
        prediction_result = model.predict(Words)
    if modelType == 'tfidf':
        if feat == 10000:
            print('Loading model')
            model = keras.models.load_model('models/tfidf_default_40000.h5')
            model.summary()
            print('\nPreprocessing test data')
            test_data, test_labels = tfidf.tf_idf_2doc(Words, labels, feat=10000)
            y_test = np.asarray(test_labels).astype('float32')
            x_test = test_data
            del test_data
            labels = test_labels
            del test_labels
            prediction_result = model.predict(x_test)
        else:
            print('Loading model')
            model = keras.models.load_model('models/tfidf_default_40000_feat3000.h5')
            model.summary()
            print('\nPreprocessing test data')
            test_data, test_labels = tfidf.tf_idf_2doc(Words, labels, feat=3000)
            y_test = np.asarray(test_labels).astype('float32')
            x_test = test_data
            del test_data
            labels=test_labels
            del test_labels
            prediction_result = model.predict(x_test)
    return prediction_result, labels


if __name__ == '__main__':
    '''
    options are:
    1 modelType='tfidf, feat=10000
    2 modelType='tfidf', feat=
    '''
    print('Loading test data')
    filename1 = 'dataset/test/neg'
    filename = 'dataset/test/pos'
    comments, targets = pre.loadData(filename, filename1)
    prediction, targets = predict(modelType='tfidf', feat=10000, Words=comments, labels=targets)
    for i in range(len(comments)):
        print('=================================================================')
        print('This is a positive comment.') if targets[i]==1 else print('This is a negative comment.')
        print('Predicted as a positive comment') if prediction[i]>0.5 else print('Predicted as a negative comment')
        print(' '.join(comments[i]))
        print('_________________________________________________________________')