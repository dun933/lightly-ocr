import torch.nn as nn
from torch.nn import *

class biLSTM(Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(biLSTM, self).__init__()
        self.rnn = LSTM(num_inputs, num_hidden, bidirectional=True)
        self.embedding = Linear(num_hidden * 2, num_outputs)

    def forward(self, inputs):
        recurent, _ = self.rnn(inputs)
        T, b, h = recurent.size()
        t_view = recurent.view(t * b, h)
        outputs = self.embedding(t_view) # -> [T*b, num_outputs]
        outputs = outputs.view(T, b, -1)

        return outputs

class CRNN(Module):
    def __init__(self, height, num_channels, num_classes, num_hidden, num_rnn=2, leaky=False):
        super(CRNN, self).__init__()
        assert height % 16 == 0, f'height should be multiple of 16, got {height} instead'

        cnn = Sequential()

        def conv2d(i, filters, kernel_size, strides, padding, bn=False):
            nIn = num_channels if i == 0 else filters[0]
            nOut = filters[1]
            cnn.add_module(f'conv_{i}', Conv2d(nIn, nOut, kernel_size, strides, padding))
            if bn:
                cnn.add_module(f'batchnorm_{i}', BatchNorm2d(nOut))
            if leaky:
                cnn.add_module(f'leaky_{i}', LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu_{i}', ReLU(inplace=True))

        conv2d(0, [num_channels, 64], 3, 1, 1)
        cnn.add_module('pooling_0', MaxPool2d(2, 2)) # -> 64x16x64
        conv2d(1, [64, 128], 3, 1, 1)
        cnn.add_module('pooling_1', MaxPool2d(2, 2)) # -> 128x8x32
        conv2d(2, [128, 256], 3, 1, 1, bn=True)
        conv2d(3, [256, 256], 3, 1, 1)
        cnn.add_module('pooling_2', MaxPool2d((2, 2), (2, 1), (0, 1))) # -> 256x4x16
        conv2d(4, [256, 512], 3, 1, 1, bn=True)
        conv2d(5, [512, 512], 3, 1, 1)
        cnn.add_module('pooling_3', MaxPool2d((2, 2), (2, 1), (0, 1))) # -> 512x2x16
        conv2d(6, [512, 512], 2, 1, 0, bn=True)

        self.cnn = cnn
        self.rnn = Sequential(biLSTM(512, num_hidden, num_hidden),
                              biLSTM(num_hidden, num_hidden, num_classes))

    def forward(self, inputs):
        conv = self.cnn(inputs)
        b, c, h, w = conv.size()
        assert h == 1, f'height of convolution layers should be 1, got {h} instead'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1) # [w,b,c]

        outputs = self.rnn(conv)
        return outputs
