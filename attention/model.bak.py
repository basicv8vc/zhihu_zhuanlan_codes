# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class LSTMMHSAN(nn.Module):
    def __init__(self, vocab_sz, i_sz, h_sz=512, h=8, dk=64, is_bidirection=True, dropout=0.3):
        super(LSTMMHSAN, self).__init__()

        self.h_sz = h_sz
        self.h = h
        self.dk = dk
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)
        self.embed = nn.Embedding(vocab_sz, i_sz) # embedding
        self.lstm = nn.LSTM(i_sz, h_sz, batch_first=True, bidirectional=is_bidirection)
        self.trans_q = nn.Linear(h_sz*2, h_sz) # 对q进行映射
        self.trans_k = nn.Linear(h_sz*2, h_sz)
        self.trans_v = nn.Linear(h_sz*2, h_sz)
        self.linear = nn.Linear(h_sz, h_sz)
        self.proj = nn.Linear(h_sz, 128)
        self.ffnn = nn.Linear(128, 4)

    def _get_mask(self, sizes):
        bh, seq_len, seq_len = sizes
        mask = np.tri(seq_len, seq_len, -1).astype(np.uint8)
        mask = 1 - mask
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0) # (1, seq_len, seq_len)

        return mask.repeat(bh, 1, 1) # (bh, seq_len, seq_len)

    def forward(self, x, is_mask=False):
        # x: (bs, seq_len)
        bs, seq_len = x.size()
        x = self.embed(x) # (bs, seq_len, i_sz)
        output, (hn, cn) = self.lstm(x) # output: (bs, seq_len, h_sz*2)
        Qt = self.trans_q(output).view(-1, seq_len, self.dk) # (bs*h, seq_len, dk)
        Kt = self.trans_k(output).view(-1, seq_len, self.dk) # (bs*h, seq_len, dk)
        Vt = self.trans_v(output).view(-1, seq_len, self.dk) # (bs*h, seq_len, dk)

        attn = torch.bmm(Qt, Kt.transpose(1, 2)) # (bs*h, seq_len, seq_len)
        if is_mask:
            mask = self._get_mask(attn.size())
            attn.masked_fill_(mask, -1e10)
        attn = self.softmax(attn)

        V = torch.bmm(attn, Vt) #(bs*h, seq_len, dk)
        V = V.view(bs, seq_len, -1) # (bs, seq_len, h*dk)
        V = self.linear(V) # (bs, seq_len, h_sz)
        V = self.dropout(V)
        V = F.leaky_relu(V)

        # average all context-aware word embedding
        V = torch.sum(V, dim=1)/seq_len #(bs, h_sz)
        V = self.proj(V)
        V = self.dropout(V)
        V = F.leaky_relu(V)
        
        return self.ffnn(V)
        


