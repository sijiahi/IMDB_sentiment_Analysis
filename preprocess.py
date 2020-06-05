import os

import re
import nltk
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords


# 数据预处理
def vectorize_sequences(sequences, dimension=10000):
    # 创建一个全0矩阵->shape(len(sequences), dimension)
    results = np.zeros((len(sequences), dimension), dtype='float16')
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def tag_words(vocabularyList):
    ptag = ['RB', 'RBR', 'RBS', 'UH', 'JJ', 'JJR', 'JJS', 'MD']  # 只选择名词和形容词
    word_list = list(vocabularyList.keys())
    tags = tqdm(nltk.pos_tag(word_list))
    for tag in tags:
        if tag[1] not in ptag:
            vocabularyList.pop(tag[0])
    return vocabularyList


# this is to load txt form data from folder
def loadData(pathDirPos, pathDirNeg):
    posAllData = []  # 积极评论
    negAllData = []  # 消极评论
    posLabels = []
    negLabels = []
    # 积极评论
    for root, dirs, files in os.walk(pathDirPos):
        print('Reading pos comment')
        for name in tqdm(files):
            # child = os.path.join('%s' % files)
            filename = os.path.join(root, name)
            lineDataPos = []  # One comment
            with open(filename, errors='ignore') as childFile:
                for lines in childFile:
                    lineString = re.sub(r'[\n\.\!\/_\-$%^*(+\"\')]+|[+—()?【】“”！:,;.？、~@#￥%…&*（）0123456789]+', ' ',
                                        lines)
                    line = lineString.split(' ')
                    for word in line:
                        if word != " " and len(word) > 1:  # 删除空白字符，并筛选出长度大于1的单词
                            lineDataPos.append(word)
            posAllData.append(lineDataPos)
            posLabels.append(1)
        print('Finished reading pos comment')
    # 消极评论
    for root, dirs, files in os.walk(pathDirNeg):
        print('Reading neg comment')
        for name in tqdm(files):
            # child = os.path.join('%s' % files)
            filename = os.path.join(root, name)
            lineDataNeg = []  # One comment
            with open(filename, errors='ignore') as childFile:
                for lines in childFile:
                    lineString = re.sub(r'[\n\.\!\/_\-$%^*(+\"\')]+|[+—()?【】“”！:,;.？、~@#￥%…&*（）0123456789]+', ' ',
                                        lines)
                    # 用空白分割/txt_sentoken/pos/" + child regEx = re.compile(r'[^a-zA-Z]|\d')
                    line = lineString.split(' ')
                    for word in line:
                        if word != " " and len(word) > 1:  # 删除空白字符，并筛选出长度大于1的单词
                            lineDataNeg.append(word)
            negAllData.append(lineDataNeg)
            negLabels.append(0)
        print('Finished reading neg comment')
        allData = posAllData + negAllData
        labels = posLabels + negLabels
    return allData, labels


def conf_data(x_train, num_words=None, skip_top=0,
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

    xs = x_train

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

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
    x_train = np.array(xs[:idx])

    return x_train


def setOfWordsToVecTor(vocabularylist, comment):
    """
    SMS内容匹配预料库，标记预料库的词汇出现的次数
    :param vocabularyList:
    :param smsWords:
    :return:
    """
    encoded_review = []
    for word in comment:
        if word in vocabularylist:
            encoded_review.append(vocabularylist.get(word))
    return encoded_review


def setOfWordsListToVecTor(vocabularylist, comments):
    """
    将文本数据的二维数组标记
    :param vocabularyList:
    :param smsWordsList:
    :return:
    """
    vocabMarkedList = []
    for i in tqdm(range(len(comments))):
        vocabMarked = setOfWordsToVecTor(vocabularylist, comments[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList


