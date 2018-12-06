# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import time
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

save_path = '../data_new/'
with open(save_path + 'sr_word2id.pkl', 'rb') as inp:
    sr_id2word = pickle.load(inp)
    sr_word2id = pickle.load(inp)
dict_word2id = dict()
for i in xrange(len(sr_word2id)):
    dict_word2id[sr_word2id.index[i].decode('utf-8')] = sr_word2id.values[i]


def get_id(word):
    """获取 word 所对应的 id.
    如果该词不在词典中，用 <UNK>（对应的 ID 为 1 ）进行替换。
    """
    if word not in dict_word2id:
        return 1
    else:
        return dict_word2id[word]


def get_id4words(words):
    """把 words 转为 对应的 id"""
    words = words.decode('utf-8')
    words = words.strip().split('|')  # 先分开词
    ids = map(get_id, words)  # 获取id
    return ids


def test_word2id():
    """把测试集的所有词转成对应的id。"""
    time0 = time.time()
    print('Processing test data.')
    path = os.getcwd()+'/test_data_without_label_seg.csv'
    df_test = pd.read_csv(path)

    df_test_question_id = df_test['question_id']
    df_test_question_title = df_test['question_title']
    df_test_question_detail = df_test['question_detail']
    save_path = '../data_new/'
    print('test question number = %d ' % len(df_test))


    # 转为 id 形式
    p = Pool()
    test_title = np.asarray(p.map(get_id4words, df_test_question_title.values))
    print(test_title)
    print(test_title.shape)
    np.save('../data_new/wd_test_title.npy', test_title)
    test_content = np.asarray(p.map(get_id4words, df_test_question_detail.values))
    print(test_content.shape)
    np.save('../data_new/wd_test_content.npy', test_content)
    p.close()
    p.join()
    print('Finished changing the test words to ids. Costed time %g s' % (time.time() - time0))


def train_word2id():
    """把训练集的所有词转成对应的id。"""
    time0 = time.time()
    print('Processing train data.')
    path = os.getcwd()+'/dev_data_seg.csv'
    df_train = pd.read_csv(path)
    #small_train = df_train.head(2)
    #df_train = small_train

    df_train_question_id = df_train['question_id']
    df_train_question_title = df_train['question_title']
    print (df_train_question_title)
    df_train_question_detail = df_train['question_detail']
    save_path = '../data_new/'
    print('train question number = %d ' % len(df_train))


    # 转为 id 形式
    p = Pool()
    train_title = np.asarray(p.map(get_id4words, df_train_question_title.values))
    print (train_title)
    print (train_title.shape)
    np.save('../data_new/wd_dev_title.npy', train_title)
    
    train_content = np.asarray(p.map(get_id4words, df_train_question_detail.values))
    print (train_content.shape)
    np.save('../data_new/wd_dev_content.npy', train_content)
    p.close()
    p.join()
    print('Finished changing the training words to ids. Costed time %g s' % (time.time() - time0))


if __name__ == '__main__':
    test_word2id()
    #train_word2id()
