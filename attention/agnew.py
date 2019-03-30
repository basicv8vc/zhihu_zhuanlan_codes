# encoding:utf-8
from io import open
import string

from torch.utils.data import DataLoader, Dataset
import numpy as np

class AGNewDataset(Dataset):
    def __init__(self, f_dir='./data/'):
        self.f_dir = f_dir
        self._initialize(self.f_dir)

    def __getitem__(self, index):
        text, label = self.train[index]
        return np.array(text, dtype=np.int64), np.array(label, dtype=np.int64)
        
    def __len__(self):
        return self.len

    def _initialize(self, f_dir):
        self.label2id = dict() # map label to id
        with open(f_dir+'classes.txt', encoding='utf-8') as fi:
            for line in fi:
                self.label2id[line.strip()] = len(self.label2id)
        # a.translate(str.maketrans('', '', string.punctuation))
        self.word2id = dict() # map word to id
        self.word2id['<PAD>'] = 0
        self.word2id['<UNK>'] = 1
        self.train_data = [] # 训练集数据
        with open(f_dir+'train_texts.txt', encoding='utf-8') as fi:
            for line in fi:
                s = line.strip()
                s = s.translate(str.maketrans('', '', string.punctuation))
                data = []
                for word in s.split():
                    if word.lower() not in self.word2id:
                        self.word2id[word.lower()] = len(self.word2id)
                    data.append(self.word2id[word.lower()])
                self.train_data.append(data)
        # add label
        self.train_labels = []
        with open(f_dir+'train_labels.txt', encoding='utf-8') as fi:
            for line in fi:
                label_id = self.label2id[line.strip()]
                self.train_labels.append(label_id)
        self.len = len(self.train_data)
        self.train = []
        for index in range(len(self.train_data)):
            self.train.append((self.train_data[index], self.train_labels[index]))
        
        self.test_data = [] # 测试集数据
        with open(f_dir+'test_texts.txt', encoding='utf-8') as fi:
            for line in fi:
                s = line.strip()
                s = s.translate(str.maketrans('', '', string.punctuation))
                data = []
                for word in s.split():
                    idx = self.word2id.get(word.lower(), 1)
                    data.append(idx)
                self.test_data.append(data)
        self.test_labels = []
        with open(f_dir+'test_labels.txt', encoding='utf-8') as fi:
            for line in fi:
                label_id = self.label2id[line.strip()]
                self.test_labels.append(label_id)
        print(len(self.test_data))
        self.test = []
        for index in range(len(self.test_data)):
            self.test.append((self.test_data[index], self.test_labels[index]))


                    



