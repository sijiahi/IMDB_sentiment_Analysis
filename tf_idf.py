import re
import pickle
from tqdm import tqdm
from nltk.stem.snowball import EnglishStemmer
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocess as pre
import numpy as np


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
                        if word != "" and len(word) > 1:  # 删除空白字符，并筛选出长度大于1的单词
                            lineDataNeg.append(word)
            negAllData.append(lineDataNeg)
            negLabels.append(0)
        print('Finished reading neg comment')
        allData = posAllData + negAllData
        labels = posLabels + negLabels
    return allData, labels


def tf_idf(comments):
    commentList = list2str(comments)
    print('Loading Vectorizer')
    if os.path.exists('models/vectorizer_imdb_tfidf.pkl'):
        with open('models/vectorizer_imdb_tfidf.pkl', 'rb') as fw:
            Vectorizer = pickle.load(fw)
        fw.close()
    else:
        print('reading data')
        filename = 'dataset/aclImdb/train/pos'
        filename1 = 'dataset/aclImdb/train/neg'
        trainComments, labels = pre.loadData(filename, filename1)
        trainCommentList = list2str(trainComments)
        Vectorizer = TfidfVectorizer(ax_features=10000, input='content', analyzer=stemmed_words,
                                     stop_words='english', encoding='utf-8', decode_error='ignore',
                                     lowercase=True, ngram_range=(1, 1))
        Vectorizer.fit_transform(trainCommentList)
        with open('models/vectorizer_imdb_tfidf.pkl', 'wb') as fw:
            pickle.dump(Vectorizer, fw)
        fw.close()
    print('Vectorizing comments')
    return Vectorizer.transform(commentList)


def tf_idf_advanced(comments):
    commentList = list2str(comments)
    print('Loading Vectorizer')
    if os.path.exists('models/vectorizer_imdb_tfidf_advanced.pkl'):
        with open('models/vectorizer_imdb_tfidf_advanced.pkl', 'rb') as fw:
            Vectorizer = pickle.load(fw)
        fw.close()
    else:
        print('reading data')
        filename = 'dataset/aclImdb/train/pos'
        filename1 = 'dataset/aclImdb/train/neg'
        trainComments, labels = pre.loadData(filename, filename1)
        trainCommentList = list2str(trainComments)
        Vectorizer = TfidfVectorizer(max_features=10000, input='content', analyzer=stemmed_words,
                                     stop_words='english'
                                     , encoding='utf-8', decode_error='ignore',
                                     lowercase=True, ngram_range=(1, 3))
        Vectorizer.fit_transform(trainCommentList)
        with open('models/vectorizer_imdb_tfidf_advanced.pkl', 'wb') as fw:
            pickle.dump(Vectorizer, fw)
        fw.close()
    print('Vectorizing comments')
    return Vectorizer.transform(commentList)


stemmer = EnglishStemmer()
analyzer = TfidfVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


def list_to_str(a_list):
    return " ".join(list(map(str, a_list)))


def list2str(comments):
    commentList = []
    for i in range(len(comments)):
        commentList.append(list_to_str(comments[i]))
    return commentList


def tf_idf_2doc(comments, labels, feat=10000):
    commentList = list2str(comments)
    rng = np.random.RandomState(seed=3)
    indices = np.arange(len(commentList))
    rng.shuffle(indices)
    commentarray = np.array(commentList)
    labelarray = np.array(labels)
    commentList = commentarray[indices]
    labels = labelarray[indices]
    commentList = commentList.tolist()
    labels = labels.tolist()
    print('Loading Vectorizer')
    if feat == 10000:
        if os.path.exists('models/vectorizer_imdb_tfidf_2doc.pkl'):
            with open('models/vectorizer_imdb_tfidf_2doc.pkl', 'rb') as fw:
                Vectorizer = pickle.load(fw)
            fw.close()
        else:
            print('reading data')
            filename = 'dataset/aclImdb/train/pos'
            filename1 = 'dataset/aclImdb/train/neg'
            trainComments, labels = pre.loadData(filename, filename1)
            trainComments = list2str(trainComments)
            trainCommentList = [list_to_str(trainComments[0:12499]), list_to_str(trainComments[12500:24999])]
            Vectorizer = TfidfVectorizer(max_features=10000, input='content', analyzer=stemmed_words,
                                         stop_words='english'
                                         , encoding='utf-8', decode_error='ignore',
                                         lowercase=True, ngram_range=(1, 3))
            Vectorizer.fit_transform(trainCommentList)
            with open('models/vectorizer_imdb_tfidf_2doc.pkl', 'wb') as fw:
                pickle.dump(Vectorizer, fw)
            fw.close()
    if feat == 3000:
        if os.path.exists('models/vectorizer_imdb_tfidf_2doc_feat3000.pkl'):
            with open('models/vectorizer_imdb_tfidf_2doc_feat3000.pkl', 'rb') as fw:
                Vectorizer = pickle.load(fw)
            fw.close()
        else:
            print('reading data')
            filename = 'dataset/aclImdb/train/pos'
            filename1 = 'dataset/aclImdb/train/neg'
            trainComments, labels = pre.loadData(filename, filename1)
            trainComments = list2str(trainComments)
            trainCommentList = [list_to_str(trainComments[0:12499]), list_to_str(trainComments[12500:24999])]
            Vectorizer = TfidfVectorizer(max_features=3000, input='content', analyzer=stemmed_words,
                                       stop_words='english'
                                       , encoding='utf-8', decode_error='ignore',
                                       lowercase=True, ngram_range=(1, 3))
            Vectorizer.fit_transform(trainCommentList)
            with open('models/vectorizer_imdb_tfidf_2doc_feat3000.pkl', 'wb') as fw:
                pickle.dump(Vectorizer, fw)
            fw.close()
    print('Vectorizing comments')
    return Vectorizer.transform(commentList), labels
