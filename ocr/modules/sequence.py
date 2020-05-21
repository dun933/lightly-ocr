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
            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
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
