#!/usr/bin/env python
# coding=utf-8
import numpy as np

# Hyper parameters
num_epoches = 10
learning_rate = 1e-3


def process_data(fileName):
    """
        处理训练数据，得到词典及词频
    """
    fi = open(fileName)
    data = fi.readlines()
    voc = dict() # 词: 词频
    for sentence in data:
        for word in sentence.split():
            voc[word] = voc.get(word, 0) + 1
    print("词典大小V={}".format(len(voc)))

    return voc

def train():
    """
        
    """
    



if __name__=='__main__':
    voc = process_data('data.txt')







