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
print('Loading test data')
filename1 = 'dataset/aclimdb/test/neg'
filename = 'dataset/aclimdb/test/pos'
Comments, labels = pre.loadData(filename, filename1)


def analyzeModel(modelType='tfidf', feat=10000, Word=Comments, labels=labels):
    if modelType == 'BOW':
        print('Loading model')
        model = keras.models.load_model('models/BOW_default_40000.h5')
        model.summary()
        fhis = open('report/BOW_default_40000.txt', 'rb')
        training_detail(fhis)
        fhis.close()
        word_index = imdb.get_word_index()
        print('\nPreprocessing data')
        Words = pre.setOfWordsListToVecTor(word_index, Word)
        Words = pre.conf_data(x_train=Words, num_words=10000)
        Words = pre.vectorize_sequences(Words)
        evaluation_result = model.evaluate(Words,
                                           labels,
                                           batch_size=512)
        prediction_result = model.predict(Words)
    if modelType == 'tfidf':
        if feat == 10000:
            print('Loading model')
            model = keras.models.load_model('models/tfidf_default_40000.h5')
            model.summary()
            fhis = open('report/tfidf_default_40000.txt', 'rb')
            training_detail(fhis)
            fhis.close()
            print('\nPreprocessing test data')
            test_data, test_labels = tfidf.tf_idf_2doc(Word, labels, feat=10000)
            y_test = np.asarray(test_labels).astype('float32')
            x_test = test_data
            del test_data
            evaluation_result = model.evaluate(x_test,
                                               y_test,
                                               batch_size=512)
            prediction_result = model.predict(x_test)
        else:
            print('Loading model')
            model = keras.models.load_model('models/tfidf_default_40000_feat3000.h5')
            model.summary()
            fhis = open('report/tfidf_default_40000_feat3000.txt', 'rb')
            training_detail(fhis)
            fhis.close()
            print('\nPreprocessing test data')
            test_data, test_labels = tfidf.tf_idf_2doc(Word, labels, feat=3000)
            y_test = np.asarray(test_labels).astype('float32')
            x_test = test_data
            del test_data
            evaluation_result = model.evaluate(x_test,
                                               y_test,
                                               batch_size=512)
            prediction_result = model.predict(x_test)
    return evaluation_result, prediction_result


def analyze(prediction, labels):
    prediction = np.ceil((prediction - 0.5))
    prediction = prediction.astype(np.int)
    prediction == labels
    prediction = prediction.tolist()
    ornot = [prediction[i][0] - labels[i] for i in range(len(labels))]
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
    print('Accuracy', (len(prediction)-(posError + negError)) / len(prediction))

def training_detail(fhis):
    history = pickle.load(fhis)
    # figure show
    # "bo" is for "blue dot"
    epochs = range(1, len(history['accuracy']) + 1)
    plt.plot(epochs, history['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history['accuracy'], 'ro', label='Training acc')
    # b is for "solid blue line"
    plt.plot(epochs, history['val_loss'], 'b', label='Validation loss')
    plt.plot(epochs, history['val_accuracy'], 'r', label='Validation acc')
    plt.title('Loss and Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Acc')
    plt.legend()
    plt.grid()
    plt.show()


def visuize_prediction(prediction, target):
    prediction = prediction.tolist()
    indices = range(1, len(prediction) + 1)
    plt.plot(indices, prediction, 'ro', markersize=1.0, label='Data Tag')
    plt.plot(indices, target, 'bo', markersize=1.0, label='Data Tag')

if __name__ == '__main__':
    '''
    options are:
    1 modelType='tfidf, feat=10000
    2 modelType='tfidf', feat=
    '''
    evaluation, prediction = analyzeModel(modelType='tfidf', feat=10000)
    analyze(prediction=prediction, labels=labels)
    visuize_prediction(prediction, labels)
