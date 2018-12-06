# -*- coding:utf-8 -*- 
import numpy as np
import pandas as pd
import word2vec
import pickle
import os
from tqdm import tqdm
import collections
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

SPECIAL_SYMBOL = ['<PAD>', '<EOS>']  # add these special symbols to word(char) embeddings.

#处理缺省值
def read_csv():

	path = os.getcwd()+'/dev_data.csv'
	df_train = pd.read_csv(path)
	headdata = df_train.head(5)
	print(headdata)
	print(df_train.shape)
	#保存csv
	headdata.to_csv('smal.csv',index=False)

'''
	#topic2id.csv:无缺省，df_topic.shape=(25551, 3)
	path = os.getcwd()+'/topic_2_id_new.csv'
	df_topic = pd.read_csv(path)
	df_topic_name = df_topic['topic_name']
	df_topic_id = df_topic['topic_id']
	df_topic_fre = df_topic['topic_frequence']

	he=df_topic_fre.sum()
	print(he)

	weight = []
	for i in range(len(df_topic_fre)):
		p = df_topic_fre.ix[i]
		weight.append(1.0-(float(p)/float(he)))
	print(weight)

	headdata = df_topic.head(5)
	print(headdata)
	#taildata = data.tail(5)
	#print (taildata)
	#describe统计下数据量、标准值、平均值、最大值等
'''	

'''
#train_data.csv：the number of train questions:721608 question_detail有173964个NaN，df_train.shape=(721608, 5),df_train:dataframe, question_title:series
#test_data_without_label.csv: the number of test questions:20596 question_detail有5215个NaN, df_test.shape=(20596, 3)
#dev_data.csc: the number of test questions:8946 question_detail有2228个NaN, df_dev.shape=(8946, 5)
	path = os.getcwd()+'/test_data_without_label.csv'
	df_train = pd.read_csv(path)
	headdata = df_train.head(2)
	print(headdata)
	print(df_train.shape)

	#查看缺失值
	df_train_question_id = df_train['question_id']
	df_train_question_title = df_train['question_title']
	df_train_question_detail = df_train['question_detail']
	#df_train_tag_id = df_train['tag_ids']
	#df_train_tag_title = df_train['tag_names']
	na_ids = df_train_question_id[df_train_question_id.isnull().values==True]
	na_title = df_train_question_title[df_train_question_title.isnull().values==True]
	na_detail = df_train_question_detail[df_train_question_detail.isnull().values==True]
	#na_tag_ids = df_train_tag_id[df_train_tag_id.isnull().values==True]
	#na_tag_names = df_train_tag_title[df_train_tag_title.isnull().values==True]

	print ('the number of test questions:%d ' %len(df_train))
	print ('df_test_question_ids.na:',len(na_ids))
	print ('df_test_question_title.na:',len(na_title))
	print ('df_test_question_detail.na:',len(na_detail))
	#print ('df_test_tag_ids.na:',len(na_tag_ids))
	#print ('df_test_tag_title.na:',len(na_tag_names))

	#取smaller data测试
	#small_train = df_train.head(1000)
	#print ('the number of smaller train questions:%d ' %len(small_train))


	# 没有 detail 的问题用 title 来替换
	na_detail_indexs = list()
	for i in tqdm(xrange(len(df_train))):
		detail = df_train.question_detail.values[i]
		if type(detail) is float:
			na_detail_indexs.append(i)
	print('There are %d test questions without detail.' % len(na_detail_indexs))
	for na_index in tqdm(na_detail_indexs):
		df_train.at[na_index, 'question_detail'] = df_train.at[na_index, 'question_title']

	#再次统计缺失值
	na_ids = df_train_question_id[df_train_question_id.isnull().values==True]
	na_title = df_train_question_title[df_train_question_title.isnull().values==True]
	na_detail = df_train_question_detail[df_train_question_detail.isnull().values==True]
	#na_tag_ids = df_train_tag_id[df_train_tag_id.isnull().values==True]
	#na_tag_names = df_train_tag_title[df_train_tag_title.isnull().values==True]

	print ('the number of test questions:%d ' %len(df_train))
	print ('df_test_question_ids.na:',len(na_ids))
	print ('df_test_question_title.na:',len(na_title))
	print ('df_test_question_detail.na:',len(na_detail))
	#print ('df_test_tag_ids.na:',len(na_tag_ids))
	#print ('df_test_tag_title.na:',len(na_tag_names))
	#保存csv
	df_train.to_csv('test_data_without_label1.csv')



	#没有 tag 的数据丢弃
	na_tag_indexs = list()
	for i in xrange(len(df_train)):
		tag_names = df_train.tag_names.values[i]
		tag_ids = df_train.tag_ids.values[i]
		if type(tag_names) is float:
			na_tag_indexs.append(i)
		if type(tag_ids) is float:
			na_tag_indexs.append(i)
	print('There are %d train questions without tag.' % len(na_tag_indexs))
	df_train = df_train.drop(na_tag_indexs)
	print('After dropping, training question number = %d' % len(df_train))

	#求平均字符长度
	data_len = pd.DataFrame(columns=["len_qt","len_qd"])
	for i in xrange(len(df_train)):
		len_qt = df_train.question_title.values[i].decode('utf-8')
		data_len.loc[i, 'len_qt'] = len(len_qt)
		#small_train['len_qt'] = small_train['question_title'].decode('utf-8').apply(len)
		len_qd = df_train.question_detail.values[i].decode('utf-8')
		data_len.loc[i, 'len_qd'] = len(len_qd)
	print data_len.describe()

#topic2id.csv:无缺省，df_topic.shape=(25551, 3)
	path = os.getcwd()+'/topic_2_id_new.csv'
	df_topic = pd.read_csv(path)
	df_topic_name = df_topic['topic_name']
	df_topic_id = df_topic['topic_id']
	df_topic_fre = df_topic['topic_frequence']

	headdata = df_topic.head(5)
	print(headdata)
	#taildata = data.tail(5)
	#print (taildata)
	#describe统计下数据量、标准值、平均值、最大值等
	print('topic_2_id:',df_topic_name.describe())
'''




#建立字典
def _build_vocab():
	path = os.getcwd()+'/train_data_seg1.csv'
	df_train = pd.read_csv(path)
	#small_train = df_train.head(1000)
	#df_train = small_train

	char=[]

	for i in tqdm(xrange(len(df_train))):
		title = df_train.loc[i,'question_title'].decode('utf-8')
		title = title.split('|')
		detail = df_train.loc[i,'question_detail'].decode('utf-8')
		detail = detail.split('|')
		char.extend(title)
		char.extend(detail)


	#print type(char)
	#print type(char)=string "如何。。。"

	#char = char.decode('utf-8')
	#print (char)
	

	counter_c=collections.Counter(char)
	count_pairs_c = sorted(counter_c.items(), key=lambda x: (-x[1], x[0]))
	charlist, _ = list(zip(*count_pairs_c))
	#print (charlist) = tuple (u' ', u'\uff0c')
	charlist = list(charlist)
	#print (charlist) = list [u' ', u'\uff0c']
	sr_id2char = pd.Series(charlist, index=range(2, 2 + len(charlist)))
	sr_char2id = pd.Series(range(2, 2 + len(charlist)), index=charlist)

	#添加特殊符号
	n_special_sym = len(SPECIAL_SYMBOL)
	for i in range(n_special_sym):
		sr_id2char[i] = SPECIAL_SYMBOL[i]
		sr_char2id[SPECIAL_SYMBOL[i]] = i
	#保存
	save_path = '../data/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	with open(save_path + 'sr_word2id.pkl', 'wb') as outp:
		pickle.dump(sr_id2char, outp)
		pickle.dump(sr_char2id, outp)


def adjust_csv():
	path = os.getcwd()+'/train_data_seg.csv'
	df_train = pd.read_csv(path)
	#保存csv
	df_train[['question_id', 'question_title','question_detail','tag_names','tag_ids']].to_csv('train_data_seg1.csv')

#求词的平均长度
def cal():
	path = os.getcwd()+'/train_data_seg1.csv'
	df_train = pd.read_csv(path)
	#small_train = df_train.head(2)
	#df_train = small_train
	print ('the number of train questions:%d ' %len(df_train))
	data_len = pd.DataFrame(columns=["len_topic"])
	for i in tqdm(xrange(len(df_train))):
		len_qt = df_train.at[i,'tag_ids']
		#print len_qt
		count_qt=1
		for q in len_qt:
			if q=='|':
				count_qt+=1
		data_len.loc[i, 'len_topic'] = count_qt
		'''
		len_qt = df_train.at[i,'question_title']
		count_qt=1
		for q in len_qt:
			if q=='|':
				count_qt+=1
		data_len.loc[i, 'len_qt'] = count_qt
		len_qd = df_train.at[i,'question_detail']
		count_qd=1
		for q in len_qd:
			if q=='|':
				count_qd+=1
		data_len.loc[i, 'len_qd'] = count_qd
		'''
	print data_len.describe()
	data_len.to_csv("topic_len.csv")

def get_word_embedding():
    """提取词向量，并保存至 ../data/word_embedding.npy"""
    print('getting the word_embedding.npy')
    #通过word2vec工具读取词向量
    wv = word2vec.load('../../wang/newsblogbbs_200_vec.txt')
    word_embedding = wv.vectors
    words = wv.vocab
    sr_id2word = pd.Series(words, index=range(1, 1 + len(words)))
    sr_word2id = pd.Series(range(1, 1 + len(words)), index=words)
    print("test"+str(len(sr_word2id)))
    print (sr_id2word)
    # 添加特殊符号：<PAD>:0, <UNK>:1
    embedding_size = 200
    n_special_sym = len(SPECIAL_SYMBOL)#2
    vec_special_sym = np.random.randn(n_special_sym, embedding_size)
    for i in range(n_special_sym):
        sr_id2word[i] = SPECIAL_SYMBOL[i]
        sr_word2id[SPECIAL_SYMBOL[i]] = i

    word_embedding = np.vstack([vec_special_sym, word_embedding])
    print("test"+str(len(word_embedding)))
    # 保存词向量
    #save_path = '../data/'
    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)
    #np.save(save_path + 'word_embedding.npy', word_embedding)

    # 保存词与id的对应关系
    #with open('sr_word2id.pkl', 'wb') as outp:
    #    pickle.dump(sr_id2word, outp)
    #    pickle.dump(sr_word2id, outp)
    #print('Saving the word_embedding.npy to ../data/word_embedding.npy')
def readindex():
	#f = open(r'../team_wang/dataset/index_file/train.index','rb')#577288
	train_index = []
	dev_index = []
	test_index = []
	index = []
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

	index = train_index + dev_index + test_index
	index = map(eval,index)
	index = np.array(index)
	for i in index:
		if i == 0:
			print i,'--exist 0'

	train_index = map(eval,train_index)
	train_index = np.array(train_index)

	dev_index = map(eval,dev_index)
	dev_index = np.array(dev_index)

	test_index = map(eval,test_index)
	test_index = np.array(test_index)

	print len(index)
	print type(index)
	print type(index[0])
	print index

	print len(train_index)
	print type(train_index)
	print type(train_index[0])
	print train_index

	print len(dev_index)
	print type(dev_index)
	print type(dev_index[0])
	print dev_index

	print len(test_index)
	print type(test_index)
	print type(test_index[0])
	print test_index

	train_num=len(train_index)
	dev_num=len(dev_index)
	test_num=len(test_index)

	train_index = index[:train_num]
	dev_index = index[train_num:train_num+dev_num]
	test_index = index[train_num+dev_num:]
	print len(train_index)
	print type(train_index)
	print type(train_index[0])
	print train_index

	print len(dev_index)
	print type(dev_index)
	print type(dev_index[0])
	print dev_index

	print len(test_index)
	print type(test_index)
	print type(test_index[0])
	print test_index



if __name__ == '__main__':
    read_csv()







