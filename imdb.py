# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:19:25 2020

@author: ASUS
"""
import re
import sys

import nltk
from tqdm import tqdm
from keras.preprocessing.sequence import _remove_long_seq
import numpy as np
import json
import warnings
import os

"""IMDB sentiment classification dataset.
"""


def load_data(word_index=None, path='imdb', num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, **kwargs):
    """Loads the IMDB dataset.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
        num_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occurring words
            (which may not be informative).
        maxlen: sequences longer than this will be filtered out.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `num_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case `maxlen` is so low
            that no input sequence could be kept.

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    """
    # Legacy support
    if 'nb_words' in kwargs:
        warnings.warn('The `nb_words` argument in `load_data` '
                      'has been renamed `num_words`.')
        num_words = kwargs.pop('nb_words')
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
    if path == 'imdb':
        if os.path.exists(os.path.join(os.getcwd(), 'dataset/imdb.npz')):
            path = os.path.join(os.getcwd(), 'dataset/imdb.npz')
        if os.path.exists(os.path.join(os.getcwd(), '../dataset/imdb.npz')):
            path = os.path.join(os.getcwd(), '../dataset/imdb.npz')
        if os.path.exists(os.path.join(os.getcwd(), 'IMDB_sentiment_Analysis/dataset/imdb.npz')):
            path = os.path.join(os.getcwd(), 'IMDB_sentiment_Analysis/dataset/imdb.npz')
    if path == 'filtered_imdb':
        if os.path.exists(os.path.join(os.getcwd(), 'dataset/filtered_imdb.npz')):
            path = os.path.join(os.getcwd(), 'dataset/filtered_imdb.npz')
        if os.path.exists(os.path.join(os.getcwd(), '../dataset/filtered_imdb.npz')):
            path = os.path.join(os.getcwd(), '../dataset/filtered_imdb.npz')
        if os.path.exists(os.path.join(os.getcwd(), 'IMDB_sentiment_Analysis/dataset/filtered_imdb.npz')):
            path = os.path.join(os.getcwd(), 'IMDB_sentiment_Analysis/dataset/filtered_imdb.npz')

    with np.load(path, allow_pickle=True) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    ## Filtering dataset with vocabulary list
    if word_index is not None:
        word_list = list(word_index.values())
        for train in tqdm(x_train):
            for word in train:
                if word not in word_list:
                    train.remove(word)
        for test in tqdm(x_test):
            for test_word in test:
                if test_word not in word_list:
                    test.remove(test_word)

    # Randomize Data
    rng = np.random.RandomState(seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    rng.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                                           'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
              for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words]
              for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def get_filtered_word_index(path='imdb_word_index.json'):
    """Retrieves the dictionary mapping words to word indices.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).

    # Returns
        The word index dictionary.
    """
    if os.path.exists(os.path.join(os.getcwd(), 'dataset/imdb_filtered_word_index.json')):
        path = os.path.join(os.getcwd(), 'dataset/imdb_filtered_word_index.json')
    if os.path.exists(os.path.join(os.getcwd(), '../dataset/imdb_filtered_word_index.json')):
        path = os.path.join(os.getcwd(), '../dataset/imdb_filtered_word_index.json')
    if os.path.exists(os.path.join(os.getcwd(), 'IMDB_sentiment_Analysis/dataset/imdb_filtered_word_index.json')):
        path = os.path.join(os.getcwd(), 'IMDB_sentiment_Analysis/dataset/imdb_filtered_word_index.json')
    with open(path) as f:
        return json.load(f)


def get_word_index(path='imdb_word_index.json'):
    """Retrieves the dictionary mapping words to word indices.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).

    # Returns
        The word index dictionary.
    """
    if os.path.exists(os.path.join(os.getcwd(), 'dataset/imdb_word_index.json')):
        path = os.path.join(os.getcwd(), 'dataset/imdb_word_index.json')
    if os.path.exists(os.path.join(os.getcwd(), '../dataset/imdb_word_index.json')):
        path = os.path.join(os.getcwd(), '../dataset/imdb_word_index.json')
    if os.path.exists(os.path.join(os.getcwd(), 'IMDB_sentiment_Analysis/dataset/imdb_word_index.json')):
        path = os.path.join(os.getcwd(), 'IMDB_sentiment_Analysis/dataset/imdb_word_index.json')
    with open(path) as f:
        return json.load(f)


def filterTaggedWords(commentData):
    if len(commentData) < 1:
        sys.stderr.write("err\n")
        return -1
    filtered = []
    labels = []
    # tagger = getTrainedTagger()
    for comment in tqdm(commentData):
        filtered_comment = []
        pos_result = nltk.pos_tag(comment)  # the output form is tuple
        for word, pos in pos_result:
            if pos in tags:
                filtered_comment.append(word)
        filtered.append(filtered_comment)
    return filtered
