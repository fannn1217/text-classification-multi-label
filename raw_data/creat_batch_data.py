# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool
import sys
import os
import random
from tqdm import tqdm

sys.path.append('../')
from data_helpers import pad_X30
from data_helpers import pad_X150
from data_helpers import pad_X50
from data_helpers import pad_X200
from data_helpers import pad_X20
from data_helpers import pad_X100
from data_helpers import train_batch
from data_helpers import eval_batch
from data_helpers import shuffle
from data_helpers import dropout
from data_helpers import getshuffle

""" 把所有的数据按照 batch_size(128) 进行打包。取 10万 样本作为验证集。
word_title_len = 30.
word_content_len = 150.
char_title_len = 52.
char_content_len = 300.
"""


wd_train_path = '../data_new/wd-data_all_230_s/data_train/'
wd_dev_path = '../data_new/wd-data_all_120/data_dev/'
wd_test_path = '../data_new/wd-data_all_120/data_test/'
ch_train_path = '../data_new/ch-data/data_train/'
ch_dev_path = '../data_new/ch-data/data_valid/'
ch_test_path = '../data_new/ch-data/data_test/'
paths = [wd_train_path, wd_dev_path, wd_test_path,
         ch_train_path, ch_dev_path, ch_test_path]
for each in paths:
    if not os.path.exists(each):
        os.makedirs(each)
'''
with open('../data/sr_topic2id.pkl', 'rb') as inp:
    sr_topic2id = pickle.load(inp)
dict_topic2id = dict()
for i in xrange(len(sr_topic2id)):
    dict_topic2id[sr_topic2id.index[i]] = sr_topic2id.values[i]
'''

def topics2ids(topics):
    """把 chars 转为 对应的 id"""
    topics = topics.split('|')
    topic = map(int, topics)
    #print (topics)
    #ids = map(lambda topic: dict_topic2id[topic], topics)          # 获取id
    return topic


def get_lables():
    """获取训练集所有样本的标签。注意之前在处理数据时丢弃了部分没有 title 的样本。"""
    
    path = os.getcwd()+'/dev_data1.csv'
    df_train = pd.read_csv(path)
    df_train_tag_ids = df_train['tag_ids']
    #print (df_train_tag_ids.values)
    p = Pool()
    y = p.map(topics2ids, df_train_tag_ids.values)
    p.close()
    p.join()
    return np.asarray(y)


# word 数据打包
def wd_train_get_batch(title_len=30, content_len=200, batch_size=128):
    print('loading word train_title and train_content.')
    train_title = np.load('../data_new/wd_train_title.npy')
    train_content = np.load('../data_new/wd_train_content.npy')

    y = np.load('../data_new/y_tr.npy')
    print('y.shape=', y.shape)



    print (" raw titles:",train_title[0], 'title.shape:',train_title.shape)
    print (" raw contents:",train_content[0],'contents.shape:', train_content.shape)
    print ("y:",y[0], 'y.shape:', y.shape)

    #补全和截断
    p = Pool()
    X_title = np.asarray(p.map(pad_X30, train_title))
    X_content = np.asarray(p.map(pad_X200, train_content))
    p.close()
    p.join()

    print ("padding 20 X_title:",X_title[0], "shape:",X_title.shape)
    print ("padding 100 X_contents:",X_content[0], "shape:",X_content.shape)

    #拼接
    X = np.hstack([X_title, X_content])
    sample_num = len(X)
    print('sample_num=%d' % sample_num)
    new_index = np.random.permutation(sample_num)
    X = X[new_index]
    y = y[new_index]
    print("X_train.shape:",X.shape,'y_train.shape=', y.shape)

    print('creating batch data.')
    # 打batch
    
    train_batch(X, y, wd_train_path, batch_size)
'''
    #划分验证集
    sample_num = X.shape[0]
    print ("sample_num:",sample_num)

    train_index = []
    dev_index = []
    test_index = []
    new_index = []
    with open(r'../team_wang/dataset/index_file/train.index','rb') as f1 :
        lines = f1.readlines()
        for line in lines:
            line.rstrip()
            train_index.append(line.strip('\n'))
    with open(r'../team_wang/dataset/index_file/dev.index','rb') as f2 :
        lines = f2.readlines()
        for line in lines:
            line.rstrip()
            dev_index.append(line.strip('\n'))
    with open(r'../team_wang/dataset/index_file/test.index','rb') as f3 :
        lines = f3.readlines()
        for line in lines:
            line.rstrip()
            test_index.append(line.strip('\n'))

    new_index = train_index + dev_index + test_index
    new_index = map(eval,new_index)
    new_index = np.array(new_index)

    #np.random.seed(13)
    #valid_num = 20000
    #new_index = np.random.permutation(sample_num)#type(new_index):<type 'numpy.ndarray'>: 如array([2, 1, 0, 3])
    train_num=len(train_index)
    dev_num=len(dev_index)
    test_num=len(test_index)

    X = X[new_index]
    y = y[new_index]
    
    X_train = X[:train_num]
    y_train = y[:train_num]
    X_dev = X[train_num:train_num+dev_num]
    y_dev = y[train_num:train_num+dev_num]
    X_test = X[train_num+dev_num:]
    y_test = y[train_num+dev_num:]


    print ("X_train.shape:",X_train.shape,'y_train.shape=', y_train.shape)
    print ("X_train:",X_train[0])

    print ("X_dev.shape:",X_dev.shape,'y_dev.shape=', y_dev.shape)
    print ("X_dev:",X_dev[0])
    print ("y_dev[0]:",y_dev[0])

    print ("X_test.shape:",X_test.shape,'y_test.shape=', y_test.shape)
    print ("X_test:",X_test[0])
'''
'''
    print('creating batch data.')
    # 验证集打batch
    sample_num = len(X_dev)
    print('dev_sample_num=%d' % sample_num)
    train_batch(X_dev, y_dev, wd_dev_path, batch_size)
    # 训练集打batch
    sample_num = len(X_train)
    print('train_sample_num=%d' % sample_num)
    train_batch(X_train, y_train, wd_train_path, batch_size)
    # 测试集打batch
    sample_num = len(X_test)
    print('test_sample_num=%d' % sample_num)
    train_batch(X_test, y_test, wd_test_path, batch_size)
    '''
'''
    X_titles = X_title[new_index]
    X_contents = X_content[new_index]
    y = y[new_index]

    X_titles_dev = X_titles[train_num:train_num+dev_num]
    X_contents_dev = X_contents[train_num:train_num+dev_num]
    y_dev = y[train_num:train_num+dev_num]

    X_titles_train = X_titles[:train_num]
    X_contents_train = X_contents[:train_num]
    y_train = y[:train_num]

    X_titles_test = X_titles[train_num+dev_num:]
    X_contents_test = X_contents[train_num+dev_num:]
    y_test = y[train_num+dev_num:]


    #数据增强，只对train数据
    titles=[]
    contents=[]
    for i in xrange(len(X_titles_train)):
        title = X_titles_train[i]
        content = X_contents_train[i]
        ti, co = getshuffle(title,content)
        titles.append(ti)
        contents.append(co)
    X_titles_train = np.asarray(titles)
    X_contents_train = np.asarray(contents)

    print ("enhance X_title:",X_titles_train[0], "shape:",X_titles_train.shape)
    print ("enhance X_content:",X_contents_train[0], "shape:",X_contents_train.shape)

    #拼接
    X_train = np.hstack([X_titles_train, X_contents_train])
    X_dev = np.hstack([X_titles_dev, X_contents_dev])
    X_test = np.hstack([X_titles_test, X_contents_test])
'''
'''
    X_titles = X_title[new_index]
    X_contents = X_content[new_index]
    y = y[new_index]

    X_titles_valid = X_titles[:valid_num]
    X_contents_valid = X_contents[:valid_num]
    y_valid = y[:valid_num]

    X_titles_train = X_titles[valid_num:]
    X_contents_train = X_contents[valid_num:]
    y_train = y[valid_num:]
    #数据增强，只对train数据
    titles=[]
    contents=[]
    for i in xrange(len(X_titles_train)):
        title = X_titles_train[i]
        content = X_contents_train[i]
        ti, co = getshuffle(title,content)
        titles.append(ti)
        contents.append(co)
    X_titles_train = np.asarray(titles)
    X_contents_train = np.asarray(contents)

    print ("enhance X_title:",X_titles_train[0], "shape:",X_titles_train.shape)
    print ("enhance X_content:",X_contents_train[0], "shape:",X_contents_train.shape)

    #拼接
    X_train = np.hstack([X_titles_train, X_contents_train])
    X_valid = np.hstack([X_titles_valid, X_contents_valid])
'''

def wd_test_get_batch(title_len=20, content_len=100, batch_size=128):
    eval_title = np.load('../data_new/wd_test_title.npy')
    eval_content = np.load('../data_new/wd_test_content.npy')
    print (" raw titles:",eval_title[0], 'title.shape:',eval_title.shape)
    print (" raw contents:",eval_content[0],'contents.shape:', eval_content.shape)

    p = Pool()
    X_title = np.asarray(p.map(pad_X20, eval_title))
    X_content = np.asarray(p.map(pad_X100, eval_content))
    p.close()
    p.join()
    print ("padding 20 X_title:",X_title[0], "shape:",X_title.shape)
    print ("padding 100 X_contents:",X_content[0], "shape:",X_content.shape)

    X = np.hstack([X_title, X_content])
    print("X.shape:",X.shape)
    sample_num = len(X)
    print('eval_sample_num=%d' % sample_num)
    eval_batch(X, wd_test_path, batch_size)


# char 数据打包
def ch_train_get_batch(title_len=30, content_len=150, batch_size=128):
    print('loading char train_title and train_content.')
    train_title = np.load('../data/ch_train_title.npy')
    train_content = np.load('../data/ch_train_content.npy')
    p = Pool()
    #补全和截断
    X_title = np.asarray(p.map(pad_X30, train_title))
    X_content = np.asarray(p.map(pad_X150, train_content))
    p.close()
    p.join()
    X = np.hstack([X_title, X_content])
    print('getting labels, this should cost minutes, please wait.')
    y = get_lables()
    print('y.shape=', y.shape)
    np.save('../data/y_tr.npy', y)
    #y = np.load('../data/y_tr.npy')
    # 划分验证集
    sample_num = X.shape[0]
    print (sample_num)
    np.random.seed(13)
    valid_num = 10000
    new_index = np.random.permutation(sample_num)
    X = X[new_index]
    y = y[new_index]
    X_valid = X[:valid_num]
    y_valid = y[:valid_num]
    X_train = X[valid_num:]
    y_train = y[valid_num:]
    print('X_train.shape=', X_train.shape, 'y_train.shape=', y_train.shape)
    print('X_valid.shape=', X_valid.shape, 'y_valid.shape=', y_valid.shape)
    # 验证集打batch
    print('creating batch data.')
    sample_num = len(X_valid)
    print('valid_sample_num=%d' % sample_num)
    train_batch(X_valid, y_valid, ch_valid_path, batch_size)
    # 训练集打batch
    sample_num = len(X_train)
    print('train_sample_num=%d' % sample_num)
    train_batch(X_train, y_train, ch_train_path, batch_size)

'''
def ch_test_get_batch(title_len=52, content_len=300, batch_size=128):
    eval_title = np.load('../data/ch_eval_title.npy')
    eval_content = np.load('../data/ch_eval_content.npy')
    p = Pool()
    X_title = np.asarray(p.map(pad_X52, eval_title))
    X_content = np.asarray(p.map(pad_X300, eval_content))
    p.close()
    p.join()
    X = np.hstack([X_title, X_content])
    sample_num = len(X)
    print('eval_sample_num=%d' % sample_num)
    eval_batch(X, ch_test_path, batch_size)
'''

if __name__ == '__main__':
    wd_train_get_batch()
    #wd_test_get_batch()
    #ch_train_get_batch()
    #ch_test_get_batch()
