
# coding: utf-8
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import pickle
import os
import sys
import time 


# 求 softmax
def _softmax(score):
    """对一个样本的输出类别概率进行 softmax 归一化.
    score: arr.shape=[1999].
    """
    max_sc = np.max(score)   # 最大分数
    score = score - max_sc
    exp_sc = np.exp(score)
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
        if predict_top5score[i] < 4.0/100:
            #print (predict_top5score[i])
            return i

time0 = time.time()
scores_names =[
'BIGRU_attention_all.npy',
'CNN-2layer-all.npy',
'RCNN_all.npy',
]  

weights = [0.3352, 0.3286,0.3362]


print(len(scores_names), len(weights))
print('All %d models' % len(weights))
sum_scores = np.zeros((20596, 25551), dtype=float)
scores_path = 'scores/'
for i in xrange(len(weights)):
    scores_name = scores_names[i]
    print('%d/%d, scores_name=%s' %(i+1, len(weights),scores_name))
    score = np.load(scores_path + scores_name)
    print(score.shape)
    sum_scores = sum_scores + score* weights[i]
print('sum_scores.shape=',sum_scores.shape)
scores_name = 'sum_scores.npy'
np.save(scores_name, sum_scores)#保存每个类别的分数
print('Finished , costed time %g s' % (time.time() - time0))



