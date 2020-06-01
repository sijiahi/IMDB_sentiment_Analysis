import json
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
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
word_index = pre.tag_words(word_index)
reverse_filtered_word_index = dict([(value, key) for (key, value) in word_index.items()])

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

with open('dataset/imdb_filtered_word_index.json', 'w', encoding='utf-8') as f:
    json.dump(word_index, f)

print(word_index)
# 将word_index反转，实现将整数索引到单词的映射


decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
