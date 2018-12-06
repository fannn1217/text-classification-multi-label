# -*- coding:utf-8 -*-
import math

"""知乎提供的评测方案"""

def score_eval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]
    
    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    precision = 0.0
    recall = 0.0
    fenzi = 0.0
    fenmu = 0.0
    right_label_num = 0  #总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  #在各个位置上总命中数量
    sample_num = 0    #总问题数量
    all_marked_label_num = 0    #总标签数量
    predict_label_num= [] #预测标签的长度

    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1     
        predict_label_num.append(len(predict_labels))
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:     #命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1  
                
    #print(len(predict_label_num))
        #print ("right_label_at_pos_num:",len(right_label_at_pos_num))
        #print ("predict_label_and_marked_label_list",len(predict_label_and_marked_label_list))
    #print "right_label_at_pos_num:",right_label_at_pos_num
    #print (right)
    
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        fenzi += right_num / math.log(3.0 + pos)
    for i in range(len(predict_label_num)):
        for pos in range(predict_label_num[i]):
            fenmu += 1 / math.log(3.0 + pos)
    precision = fenzi / fenmu
    recall = float(right_label_num) / all_marked_label_num
    return precision, recall, 2 * (precision * recall) / (precision + recall )
    
'''
predict_label_and_marked_label_list = [ ([4,6,2,3], [1,2,3,4,5])]
a,b,c = score_eval(predict_label_and_marked_label_list)
print a,b,c
'''