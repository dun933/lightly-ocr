import torch
import torch.nn as nn
import torch.nn.functional as tf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(biLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True) # -> [b, seq, feats]
        self.linear = nn.Linear(hidden_size*2, output_size) # -> similar to Dense in tf

    def forward(self, inputs):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(inputs) # [b, T, input_size] -> [b, T, (2*hidden_size)] since LSTM in pytorch returns hidden_state and cell state
        outputs = self.linear(recurrent)
        return outputs

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.linear = nn.Linear(hidden_size, num_classes)

    def onehot_decode(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        onehot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        return onehot.scatter_(1,input_char, 1)

    def forward(self, batch_hidden, text, is_train=True, batch_max_len = 25):
        # batch_hidden is the hidden state of lstm [batch, num_steps, num_classes]
        # text index of each image [batch_size x (max_len+1)] -> include [GO] token
        # return probability distribution of each step [batch_size, num_steps, num_classes]

        batch_size = batch_hidden.size(0)
        num_steps = batch_max_len+1
        h = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        nh = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
              torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                onehot_char = onehot_decode(text[:, i], onehot_dim=self.num_classes)
                nh, alpha = self.cell(nh, batch_hidden, onehot_char) # nh: decoder's hidden at s_(t-1), batch_hidden: encoder's hidden H, onehot_char: onehot(y_(t_1))
                h[:,i,:] = nh[0] # -> lstm hidden index (0:hidden, 1:cell)
            probs = self.linear(h)
        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                onehot_char = self.onehot_decode(targets, onehot_dim=self.num_classes)
                nh, alpha = self.cell(nh, batch_hidden, onehot_char)
                probs_step = self.linear(nh[0])
                probs[:,i,:] = probs_step
                _, n_in = probs_step.max(1) # -> returns next inputs
                targets = n_in

        return probs

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size+num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, h, batch_hidden, onehot_char):
        # [b, num_steps, num_channels] -> [b, num_steps, hidden_size]
        batch_proj = self.i2h(batch_hidden)
        h_proj = self.h2h(h[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_proj+h_proj))

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0,2,1), batch_hidden).squeeze(1) # batch x num_channels
        concat = torch.cat([context, onehot_char], 1) # batch x ( num_channels + num_embeddings )
        nh = self.rnn(concat, h)
        return nh, alpha
