# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import time
import network
import pandas as pd
from pandas.core.frame import DataFrame

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('../..')
from evaluator1 import score_eval
from data_helpers import to_categorical

settings = network.Settings()
title_len = settings.title_len
model_name = settings.model_name
ckpt_path = settings.ckpt_path

local_scores_path = '../../local_scores/'
scores_path = '../../scores/'
if not os.path.exists(local_scores_path):
    os.makedirs(local_scores_path)
if not os.path.exists(scores_path):
    os.makedirs(scores_path)

embedding_path = '../../data_new/word_embedding.npy'
data_valid_path = '../../data_new/wd-data_all_230_s/data_dev/'
data_test_path = '../../data_new/wd-data_all_230/data_test/'
#data_train_path = '../../data_new/wd-data/data_train/'
va_batches = os.listdir(data_valid_path)
te_batches = os.listdir(data_test_path)  # batch 文件名列表
n_va_batches = len(va_batches)
n_te_batches = len(te_batches)


def get_batch(batch_id):
    """get a batch from valid data"""
    new_batch = np.load(data_valid_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    X1_batch = X_batch[:, :title_len]
    X2_batch = X_batch[:, title_len:]
    return [X1_batch, X2_batch, y_batch]

def get_batch_t(batch_id):
    """get a batch from valid data"""
    new_batch = np.load(data_test_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    X1_batch = X_batch[:, :title_len]
    X2_batch = X_batch[:, title_len:]
    return [X1_batch, X2_batch, y_batch]


def get_test_batch(batch_id):
    """get a batch from test data"""
    X_batch = np.load(data_test_path + str(batch_id) + '.npy')
    X1_batch = X_batch[:, :title_len]
    X2_batch = X_batch[:, title_len:]
    return [X1_batch, X2_batch]

def get_sequence_length(X1_batch, X2_batch):
    #求sequence_length
    #X1_length = [[0 for col in range(1)] for row in range(y_batch)]
    X1_length = [0]*len(X1_batch)
    for i in range(len(X1_batch)):
        for j in X1_batch[i]:
            if  j!=0:
                X1_length[i] += 1
    #X1_length = np.array(X1_length)
    X2_length = [0]*len(X1_batch)
    for i in range(len(X2_batch)):
        for j in X2_batch[i]:
            if  j!=0:
                X2_length[i] += 1
    #X2_length = np.array(X2_length)
    return X1_length, X2_length

# # 求 softmax
def _softmax(score):
    """对一个样本的输出类别概率进行 softmax 归一化.
    score: arr.shape=[1999].
    """
    max_sc = np.max(score)   # 最大分数
    score = score - max_sc
    exp_sc = np.exp(score)  #e的score次方
    sum_exp_sc = np.sum(exp_sc)
    softmax_sc = exp_sc / sum_exp_sc
    return softmax_sc    # 归一化的结果
    
def softmax(scores):
    """对所有样本的输出概率进行 softmax 归一化处理。
    scores: arr.shape=[n_sample, 1999].
    """
    softmax_scs = map(_softmax, scores)
    return np.asarray(softmax_scs)

def findindex(predict_top5score):
    for i in range(len(predict_top5score)):
        if predict_top5score[i] < 2.0/100:
            #print (predict_top5score[i])
            return i

def tolen(x):
    return len(x)

def predict_dev(sess, model):
    """Test on the valid data."""
    time0 = time.time()
    predict_labels_list = list()  # 所有的预测结果
    predict_score20_list = list() # 预测排名前20的分数
    predict_labels_list2 = list() #前五名的结果
    marked_labels_list = list()
    topic_num = list()
    predict_scores = list()
    for i in tqdm(xrange(n_va_batches)):#验证集
        [X1_batch, X2_batch, y_batch] = get_batch(i)
        X1_length, X2_length = get_sequence_length(X1_batch, X2_batch)
        marked_labels_list.extend(y_batch)#真实标签结果 没-1
        y_batch = to_categorical(y_batch)
        _batch_size = len(X1_batch)
        fetches = [model.y_pred]#每个类别的分数
        feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch,
                     model.batch_size: _batch_size, model.X1_length: X1_length, model.X2_length: X2_length,
                     model.tst: True, model.keep_prob: 1.0}
        predict_labels = sess.run(fetches, feed_dict)[0]
        predict_labels = softmax(predict_labels)#128
        predict_scores.append(predict_labels)#每个类别的分数


        predict_top5score = map(lambda label: np.sort(label,axis=-1)[-1:-6:-1], predict_labels)  # 取最大的5个分数 128
        #predict_top20score = map(lambda label: np.sort(label,axis=-1)[-1:-21:-1], predict_labels)  # 取最大的20个分数 128
        #print (type(predict_score20_list))
        #print (type(predict_top20score))
        #predict_score20_list.extend(predict_top20score) #所有
        #list,predict_score_list1[0]=[ 0.63514245  0.09193601  0.0417341   0.02742104  0.02721145]

        index = map(findindex,predict_top5score)#list 128
        #print (index,'index.type:',type(index),'len.index',len(index))

        predict_toplabels = list()

        for i in range(len(index)):
            if index[i] == None:
                toplabel = predict_labels[i].argsort()[-1:-6:-1]
            elif index[i] == 0:
                toplabel = predict_labels[i].argsort()[-1:-2:-1]
            else:
                toplabel = predict_labels[i].argsort()[-1:-1*index[i]-1:-1]
            predict_toplabels.append(toplabel)

        predict_labels_list.extend(predict_toplabels) 
        #print('predict_toplabels:',predict_toplabels,type(predict_toplabels),len(predict_toplabels))


        #predict_top5labels = map(lambda label: label.argsort()[-1:-6:-1], predict_labels)  # 取最大的5个下标
        #predict_labels_list2.extend(predict_top5labels)
        
        #predict_labels_list2.to_csv('predict_labels_list2.csv')


    #predict_score20_list = DataFrame(predict_score20_list)
    #predict_labels_list2 = DataFrame(predict_labels_list2)
    #predict_score20_list.to_csv('score20list.csv')
    #predict_labels_list2.to_csv('predict_labels_list2.csv')
    #topic_num = map(tolen,marked_labels_list)
    #topic_num = DataFrame(topic_num)
    #topic_num.to_csv('topic_num.csv')
    predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)#都-1了 不知道为啥

    print (predict_label_and_marked_label_list[0:2])
    #(array([ 15, 327, 307, 478,  10]), [8, 15, 307, 0])，真实是[9, 16, 308, 1]
    precision, recall, f1 = score_eval(predict_label_and_marked_label_list)#计算分数
    print('Local valid p=%g, r=%g, f1=%g' % (precision, recall, f1))
    predict_scores = np.vstack(np.asarray(predict_scores))
    print('predict_scores:',predict_scores.shape)
    local_scores_name = local_scores_path + model_name + '_dev.npy'
    np.save(local_scores_name, predict_scores)#保存每个类别的分数
    print('local_scores.shape=', predict_scores.shape)
    print('Writed the dev scores into %s, time %g s' % (local_scores_name, time.time() - time0))


def predict(sess, model):
    """Test on the test data."""
    time0 = time.time()
    predict_scores = list()
    for i in tqdm(xrange(n_te_batches)):
        [X1_batch, X2_batch] = get_test_batch(i)
        X1_length, X2_length = get_sequence_length(X1_batch, X2_batch)
        _batch_size = len(X1_batch)
        fetches = [model.y_pred]
        feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch,
                     model.batch_size: _batch_size, model.X1_length: X1_length, model.X2_length: X2_length,
                     model.tst: True, model.keep_prob: 1.0}
        predict_labels = sess.run(fetches, feed_dict)[0]
        predict_labels = softmax(predict_labels)#128
        predict_scores.append(predict_labels)
    predict_scores = np.vstack(np.asarray(predict_scores))
    scores_name = scores_path + model_name + '.npy'
    np.save(scores_name, predict_scores)
    print('scores.shape=', predict_scores.shape)
    print('Writed the scores into %s, time %g s' % (scores_name, time.time() - time0))


def main(_):
    if not os.path.exists(ckpt_path + 'checkpoint'):
        print('there is not saved model, please check the ckpt path')
        exit()
    print('Loading model...')
    W_embedding = np.load(embedding_path)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.50
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        model = network.BiGRU(W_embedding, settings)
        model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        print('dev predicting...')
        predict_dev(sess, model)
        print('test predicting...')
        predict(sess, model)


if __name__ == '__main__':
    tf.app.run()
