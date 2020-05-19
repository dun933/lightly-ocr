import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FractionalPickup(nn.Module):
    def __init__(self):
        super(FractionalPickup, self).__init__()

    def forward(self, x):
        x_shape = x.size()
        assert len(x_shape) == 4
        assert x_shape[2] == 1

        frac_num = 1

        h_list = 1.
        w_list = np.arange(x_shape[3]) * 2. / (x_shape[3] - 1) - 1
        for i in range(frac_num):
            idx = int(np.random.rand() * len(w_list))
            if idx <= 0 or idx >= x_shape[3] - 1:
                continue
            beta = np.random.rand() / 4.
            value0 = (beta * w_list[idx] + (1 - beta) * w_list[idx - 1])
            value1 = (beta * w_list[idx - 1] + (1 - beta) * w_list[idx])
            w_list[idx - 1] = value0
            w_list[idx] = value1

        grid = np.meshgrid(w_list, h_list, indexing='ij')
        grid = np.stack(grid, axis=-1)
        grid = np.transpose(grid, (1, 0, 2))
        grid = np.expand_dims(grid, 0)
        grid = np.tile(grid, [x_shape[0], 1, 1, 1])
        grid = torch.from_numpy(grid).type(x.data.type()).to(device)
        self.grid = torch.Tensor(grid, requires_grad=False)

        x_offset = F.grid_sample(x, self.grid, align_corners=True)

        return x_offset


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, dropout=0.3):
        super(BidirectionalLSTM, self).__init__()
        # batch_first=True-> [b, seq, feats]
        self.dropout = dropout
        if self.dropout:
            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=dropout)
        else:
            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, inputs):
        if torch.cuda.is_available():
            self.rnn.flatten_parameters()
        # [b, T, nIn] -> [b, T, (2*nHidden)]
        recurrent, _ = self.rnn(inputs)
        if self.dropout:
            T, b, h = recurrent.size()
            t_rec = recurrent.view(T * b, h)
            outputs = self.embedding(t_rec)  # [T*b, nOut]
            outputs = outputs.view(T, b, -1)
        else:
            outputs = self.embedding(recurrent)  # returns [b, T, nOut]
        return outputs


class AttentionCell(nn.Module):
    # TODO: Fixes batch_first in LSTM to fix foward() difference when processing i2h
    def __init__(self, nIn, nHidden, num_embeddings, frac_pickup=False):
        super(AttentionCell, self).__init__()
        self.nHidden = nHidden
        self.i2h = nn.Linear(nIn, nHidden, bias=False)
        self.h2h = nn.Linear(nHidden, nHidden)
        self.score = nn.Linear(nHidden, 1, bias=False)
        self.is_frac = frac_pickup
        if frac_pickup:
            self.frac_pickup = FractionalPickup()
            self.rnn = nn.GRUCell(nIn + num_embeddings, nHidden)
        else:
            self.frac_pickup = None
            self.rnn = nn.LSTMCell(nIn + num_embeddings, nHidden)

    def forward(self, h, feats, cur_embed, test=False):
        # [b, num_steps, num_channels] -> [b, num_steps, nHidden]
        # NOTES: cur_embed is onehot_char for CRNN
        if self.frac_pickup is not None:
            # MORAN -> #TODO: use batch_first instead
            assert feats.batch_first is False, 'I will fix MORAN to use batch_first in the near future, but now feats must be shape [T x b x num_classes]'
            nT, nB, nC = feats.size()
            nHidden = self.nHidden
            feats_ = self.i2h(feats.view(-1, nC))  # feats projection
            # serves same functionality with self.h2h(h[0]).unsqueeze(1)
            h_ = self.h2h(h).view(1, nB, nHidden).expand(nT, nB, nHidden).contiguous().view(-1, nHidden)  # hidden projection
            emit = self.score(torch.tanh(feats_ + h_).view(-1, nHidden).view(nT, nB))  # fuck batch_first ngl
            alpha = F.softmax(emit, dim=0)  # nT x nB
        else:
            # uses for CRNN decoder
            feats_ = self.i2h(feats)
            h_ = self.h2h(h[0]).unsqueeze(0)
            emit = self.score(torch.tanh(feats_ + h_))
            alpha = F.softmax(emit, dim=1)  # nT x nB

        if self.is_frac:
            #TODO: fail-proof for using fractional_pickup temporary, should switch to `batch_first`
            if not test:
                # yapf: disable
                _alpha = alpha.transpose(0, 1).contiguous().unsqueeze(1)
                alpha_fp = self.frac_pickup(_alpha.unsqueeze(2)).squeeze()
                alpha_fp = alpha_fp.transpose(0, 1).contiguous().view(nT, nB, 1).expand(nT, nB, nC)
                # FIXME: feats is not batch_first -> future-proof with torch.bmm
                context = (feats * alpha_fp).sum(0).squeeze(0)  # nB * nC
                if len(context.size()) == 1:
                    context = context.unsqueeze(0)
                context = torch.cat([context, cur_embed], 1)
                nh = self.rnn(context, h)
                return nh, alpha_fp
            # FIXME: this could get pass if test=True
            else:
                context = (feats * alpha.view(nT, nB, 1).expand(nT, nB, nC)).sum(0).squeeze(0)  # nB * nC
                if len(context.size()) == 1:
                    context = context.unsqueeze(0)
                context = torch.cat([context, cur_embed], 1)
                nh = self.rnn(context, h)
                return nh, alpha
        else:
            # with batch_first -> CRNN
            # context : batch x num_channels
            context = torch.bmm(alpha.permute(0, 2, 1), feats).squeeze(1)
            # batch x ( num_channels + num_embeddings )
            concat = torch.cat([context, cur_embed], 1)
            nh = self.rnn(concat, h)
            return nh, alpha
            # yapf: enable


class Attention(nn.Module):
    def __init__(self, nIn, nHidden, num_classes):
        super(Attention, self).__init__()
        self.cell = AttentionCell(nIn, nHidden, num_classes)
        self.nHidden = nHidden
        self.num_classes = num_classes
        self.linear = nn.Linear(nHidden, num_classes)

    def onehot_decode(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        onehot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        return onehot.scatter_(1, input_char, 1)

    def forward(self, batch_hidden, text, is_train=True, batch_max_len=25):
        # batch_hidden is the hidden state of lstm [batch, num_steps, num_classes]
        # text index of each image [batch_size x (max_len+1)] -> include [GO] token
        # return probability distribution of each step [batch_size, num_steps, num_classes]

        batch_size = batch_hidden.size(0)
        num_steps = batch_max_len + 1
        h = torch.FloatTensor(batch_size, num_steps, self.nHidden).fill_(0).to(device)
        nh = (torch.FloatTensor(batch_size, self.nHidden).fill_(0).to(device), torch.FloatTensor(batch_size, self.nHidden).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                onehot_char = self.onehot_decode(text[:, i], onehot_dim=self.num_classes)
                nh, alpha = self.cell(nh, batch_hidden,
                                      onehot_char)  # nh: decoder's hidden at s_(t-1), batch_hidden: encoder's hidden H, onehot_char: onehot(y_(t_1))
                h[:, i, :] = nh[0]  # -> lstm hidden index (0:hidden, 1:cell)
            probs = self.linear(h)
        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                onehot_char = self.onehot_decode(targets, onehot_dim=self.num_classes)
                nh, alpha = self.cell(nh, batch_hidden, onehot_char)
                probs_step = self.linear(nh[0])
                probs[:, i, :] = probs_step
                _, n_in = probs_step.max(1)  # -> returns next inputs
                targets = n_in

        return probs
