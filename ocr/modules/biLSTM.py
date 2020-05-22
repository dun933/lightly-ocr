import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, dropout=None):
        super(BidirectionalLSTM, self).__init__()
        # batch_first=True-> [b, seq, feats]
        self.dropout = dropout
        if self.dropout is not None:
            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
            self.embedding = nn.Linear(nHidden * 2, nOut)
        else:
            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
            self.linear = nn.Linear(nHidden * 2, nOut)

    def forward(self, inputs):
        if torch.cuda.is_available():
            self.rnn.flatten_parameters()
        # [b, T, nIn] -> [b, T, (2*nHidden)]
        recurrent, _ = self.rnn(inputs)
        if self.dropout is not None:
            T, b, h = recurrent.size()
            t_rec = recurrent.view(T * b, h)
            outputs = self.embedding(t_rec)  # [T*b, nOut]
            outputs = outputs.view(T, b, -1)
        else:
            outputs = self.linear(recurrent)  # returns [b, T, nOut]
        return outputs


# attention layer for CRNN
class Attention(nn.Module):
    def __init__(self, nIn, nHidden, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(nIn, nHidden, num_classes)
        self.nHidden = nHidden
        self.num_classes = num_classes
        self.generator = nn.Linear(nHidden, num_classes)

    def onehot_decode(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        onehot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        return onehot.scatter_(1, input_char, 1)

    def forward(self, batch_hidden, text, training=True, batch_max_len=25):
        # batch_hidden is the hidden state of lstm [batch, num_steps, num_classes]
        # text index of each image [batch_size x (max_len+1)] -> include [GO] token
        # return probability distribution of each step [batch_size, num_steps, num_classes]
        batch_size = batch_hidden.size(0)
        num_steps = batch_max_len + 1
        h = torch.FloatTensor(batch_size, num_steps, self.nHidden).fill_(0).to(device)
        nh = (torch.FloatTensor(batch_size, self.nHidden).fill_(0).to(device), torch.FloatTensor(batch_size, self.nHidden).fill_(0).to(device))

        if training:
            for i in range(num_steps):
                onehot_char = self.onehot_decode(text[:, i], onehot_dim=self.num_classes)
                nh, alpha = self.attention_cell(
                    nh, batch_hidden, onehot_char)  # nh: decoder's hidden at s_(t-1), batch_hidden: encoder's hidden H, onehot_char: onehot(y_(t_1))
                h[:, i, :] = nh[0]  # -> lstm hidden index (0:hidden, 1:cell)
            probs = self.generator(h)
        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                onehot_char = self.onehot_decode(targets, onehot_dim=self.num_classes)
                nh, alpha = self.attention_cell(nh, batch_hidden, onehot_char)
                probs_step = self.generator(nh[0])
                probs[:, i, :] = probs_step
                _, n_in = probs_step.max(1)  # -> returns next inputs
                targets = n_in

        return probs
