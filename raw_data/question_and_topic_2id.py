# -*- coding:utf-8 -*- 

import pandas as pd
import pickle
import os


def question_and_topic_2id():
    """把question和topic转成id形式并保存至 ../data/目录下。"""
    path = os.getcwd()+'/train_data_new1.csv'
    df_train = pd.read_csv(path)
    df_train_question_id = df_train['question_id']
    save_path = '../data_new/'
    print('question number = %d ' % len(df_train_question_id))

    path = os.getcwd()+'/topic_2_id_new.csv'
    df_topic = pd.read_csv(path)
    df_topic_name = df_topic['topic_name']
    df_topic_id = df_topic['topic_id']
    print('topic number = %d ' % len(df_topic_id ))

    # 问题 id 按照给出的问题顺序编号
    questions = df_train_question_id.values
    sr_question2id = pd.Series(range(len(questions)), index=questions) 
    sr_id2question = pd.Series(questions, index=range(len(questions)))
    print 'sr_id2question finish'

    # topic 按照给定id编号
    topics = df_topic_name.values
    topics_id = df_topic_id.values
    sr_topic2id = pd.Series(topics_id,index=topics)
    sr_id2topic = pd.Series(topics, index=topics_id) 
    print 'sr_id2topic finish'

    with open(save_path + 'sr_question2id.pkl', 'wb') as outp:
        pickle.dump(sr_question2id, outp)
        pickle.dump(sr_id2question, outp)
    with open(save_path + 'sr_topic2id.pkl', 'wb') as outp:
        pickle.dump(sr_topic2id, outp)
        pickle.dump(sr_id2topic, outp)
    print('Finished changing.')


if __name__ == '__main__':
    question_and_topic_2id()
