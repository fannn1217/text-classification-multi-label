# -*- coding:utf-8 -*- 
import numpy as np
import pandas as pd
import os
import word2vec
import jieba
from tqdm import tqdm

'''
seg_list = jieba.cut("我来到清华大学",cut_all = False)
print "default mode = ", "|".join(seg_list)

data = pd.DataFrame(columns=["data"])
data.loc[0, 'data'] = "我来到清华大学"
data.loc[1, 'data'] = "我来到北京大学"
'''
def seg():
	path = os.getcwd()+'/test_data_without_label1.csv'
	df_train = pd.read_csv(path)

	#取smaller data测试
	#small_train = df_train.head(10)
	#print ('the number of smaller train questions:%d ' %len(small_train))
	print ('the number of test questions:%d ' %len(df_train))

	#分词
	for i in tqdm(xrange(len(df_train))):
		seg_title = jieba.cut(df_train.at[i,'question_title'],cut_all = False)
		df_train.at[i, 'question_title'] = "|".join(seg_title)
		seg_detail = jieba.cut(df_train.at[i,'question_detail'],cut_all = False)
		df_train.at[i, 'question_detail'] = "|".join(seg_detail)

	#保存csv
	df_train.to_csv('test_data_without_label_seg.csv',encoding='utf-8')
	print df_train.head(5)
'''
	#求平均词长度
	data_len = pd.DataFrame(columns=["len_qt","len_qd"])
	for i in tqdm(xrange(len(df_train))):
		len_qt = 0
		for w in df_train.question_title.values[i]:
			if w == '|':
				len_qt += 1
		data_len.at[i, 'len_qt'] = len_qt+1

		len_qd = 0
		for k in df_train.question_detail.values[i]:
			if k == '|':
				len_qd += 1
		data_len.at[i, 'len_qd'] = len_qd+1
	print (data_len.describe())
	print (data_len.head(5))
	data_len.to_csv("word_len_new.csv")
'''



if __name__ == '__main__':
	seg()
