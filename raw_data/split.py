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
from pandas.core.frame import DataFrame
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

time0 = time.time()
data_valid_path = '../data/wd-data_120_shuffle/data_valid/'
new_batch = np.load(data_valid_path + '0.npz')
X_batch = new_batch['X']
y_batch = new_batch['y']

print("y_batch[0]",y_batch[0])


print('Processing train data.')
path = os.getcwd()+'/train_data_seg1.csv'
df_train = pd.read_csv(path)
#small_train = df_train.head(5)
#print ("df_train:",small_train)
#df_train = small_train

print('train question number = %d ' % len(df_train))
print("df_train.type:",type(df_train))

sample_num = len(df_train)
np.random.seed(13)
valid_num = 20000
new_index = np.random.permutation(sample_num)
#print ("new_index:",new_index)

valid_index = new_index[:valid_num]
#print ("valid_index:",valid_index)
train_index = new_index[valid_num:]
valids = []
trains = []
for index in valid_index:
	valid = df_train.ix[index]
	valids.append(valid)
for index in train_index:
	train = df_train.ix[index]
	trains.append(train)

valids = DataFrame(valids)
trains = DataFrame(trains)
print ("trains.shape:",trains.shape)
#print("trains[0]:",trains[0])
print ("valids.shape:",valids.shape)
#print("valids[0]:",valids[0])
print("valid.type",type(valids))
#保存csv
valids[['question_id', 'question_title','question_detail','tag_names','tag_ids']].to_csv('split_valid.csv')
trains[['question_id', 'question_title','question_detail','tag_names','tag_ids']].to_csv('split_train.csv')
#print ("train.shape:",train.shape())


