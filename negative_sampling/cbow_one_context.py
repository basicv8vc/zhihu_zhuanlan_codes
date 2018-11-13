#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function

def process_data(file_name):
    """处理训练数据，得到词典及词频"""
    fi = open(file_name)
    data = fi.readlines()
    voc = dict() # 词: 词频
    for sentence in data:
        for word in sentence.split():
            voc[word] = voc.get(word, 0) + 1
    print("词典大小V={}".format(len(voc)))
    return voc

if __name__ == '__main__':
    vocab = process_data('data.txt')
