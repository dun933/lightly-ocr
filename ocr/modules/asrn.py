import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionCell
from .biLSTM import BidirectionalLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ASRN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, bidirectional=False, CUDA=True):
        super(ASRN, self).__init__()
        assert imgH % 16 == 0, 'imgH must be a multiple of 16'

        self.cnn = ResNet(nc)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
        )

        self.bidirectional = bidirectional
        if self.bidirectional:
            self.attentionL2R = Attention(nh, nh, nclass, 256)
            self.attentionR2L = Attention(nh, nh, nclass, 256)
        else:
            self.attention = Attention(nh, nh, nclass, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, length, text, text_rev, test=False):
        # conv features
        conv = self.cnn(inputs)

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1).contiguous()  # [w, b, c]

        # rnn features
        rnn = self.rnn(conv)

        if self.bidirectional:
            outputL2R = self.attentionL2R(rnn, length, text, test)
            outputR2L = self.attentionR2L(rnn, length, text_rev, test)
            return outputL2R, outputR2L
        else:
            outputs = self.attention(rnn, length, text, test)
        return outputs


# MORAN implementations
# TODO: just fix their code this is horrible
class Attention(nn.Module):
    def __init__(self, nIn, nHidden, num_classes, num_embeddings=128):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(nIn, nHidden, num_embeddings, MORAN=True)
        self.nIn = nIn
        self.nHidden = nHidden
        self.generator = nn.Linear(nHidden, num_classes)
        self.char_embeddings = nn.Parameter(torch.randn(num_classes + 1, num_embeddings))
        self.num_classes = num_classes

    # yapf: enable
    # targets is nT * nB
    def forward(self, feats, len_text, text, test=False):

        nIn = self.nIn
        nHidden = self.nHidden
        nT, nB, nC = feats.size()
        assert (nIn == nC) and (nB == len_text.numel())

        num_steps = len_text.data.max()
        num_labels = len_text.data.sum()

        if not test:

            targets = torch.zeros(nB, num_steps + 1).long().to(device)
            start_id = 0

            for i in range(nB):
                targets[i][1:1 + len_text.data[i]] = text.data[start_id:start_id + len_text.data[i]] + 1
                start_id = start_id + len_text.data[i]
            targets = targets.transpose(0, 1).contiguous()

            output_hidden = torch.zeros(num_steps, nB, nHidden).type_as(feats.data)
            hidden = torch.zeros(nB, nHidden).type_as(feats.data)

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets[i])
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                output_hidden[i] = hidden

            new_hidden = torch.zeros(num_labels, nHidden).type_as(feats.data)
            b = 0
            start = 0

            for length in len_text.data:
                new_hidden[start:start + length] = output_hidden[0:length, b, :]
                start = start + length
                b = b + 1

            probs = self.generator(new_hidden)
            return probs

        else:

            hidden = torch.zeros(nB, nHidden).type_as(feats.data)
            targets_temp = torch.zeros(nB).long().contiguous().to(device)
            probs = torch.zeros(nB * num_steps, self.num_classes).to(device)

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets_temp)
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                hidden2class = self.generator(hidden)
                probs[i * nB:(i + 1) * nB] = hidden2class
                _, targets_temp = hidden2class.max(1)
                targets_temp += 1

            probs = probs.view(num_steps, nB, self.num_classes).permute(1, 0, 2).contiguous()
            probs = probs.view(-1, self.num_classes).contiguous()
            probs_res = torch.zeros(num_labels, self.num_classes).type_as(feats.data)
            b = 0
            start = 0

            for length in len_text.data:
                probs_res[start:start + length] = probs[b * num_steps:b * num_steps + length]
                start = start + length
                b = b + 1

            return probs_res


class ResNet(nn.Module):
    def __init__(self, c_in):
        super(ResNet, self).__init__()
        self.block0 = nn.Sequential(nn.Conv2d(c_in, 32, 3, 1, 1), nn.BatchNorm2d(32, momentum=0.01))
        self.block1 = self._make_layer(32, 32, 2, 3)
        self.block2 = self._make_layer(32, 64, 2, 4)
        self.block3 = self._make_layer(64, 128, (2, 1), 6)
        self.block4 = self._make_layer(128, 256, (2, 1), 6)
        self.block5 = self._make_layer(256, 512, (2, 1), 3)

    def _make_layer(self, c_in, c_out, stride, repeat=3):
        layers = []
        layers.append(residual_block(c_in, c_out, stride))
        for i in range(repeat - 1):
            layers.append(residual_block(c_out, c_out, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        block0 = self.block0(x)
        block1 = self.block1(block0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        return block5


class residual_block(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(residual_block, self).__init__()
        self.downsample = None
        flag = False
        if isinstance(stride, tuple):
            if stride[0] > 1:
                self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1), nn.BatchNorm2d(c_out, momentum=0.01))
                flag = True
        else:
            if stride > 1:
                self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1), nn.BatchNorm2d(c_out, momentum=0.01))
                flag = True
        if flag:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1), nn.BatchNorm2d(c_out, momentum=0.01))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, 0), nn.BatchNorm2d(c_out, momentum=0.01))
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, 3, 1, 1), nn.BatchNorm2d(c_out, momentum=0.01))
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(residual + conv2)
