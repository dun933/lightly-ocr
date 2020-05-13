import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

class Converter(object):
    """convert string to label"""
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'

        self.char_dict = {}
        for i, char in enumerate(alphabet):
            self.char_dict[char] = i+1 # 0 for whitespace

    def encode(self, text):
        # returns encoded_text<torch.IntTensor> and length<torch.IntTensor>
        if isinstance(text, str):
            text = [self.char_dict[char.lower() if self._ignore_case else char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text),torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        # returns text converted from t and length. raw indicates whether it's a raw string or not
        if length.numel() ==1:
            length = length[0]
            assert t.numel() == length, f'length and len(text) should be equal. Got {t.numel()} for len(text) and {length} for length'
            if raw:
                return ''.join([self.alphabet[i-1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[0]!=0 and (not (i>0 and t[i-1]==t[i])):
                        char_list.append(self.alphabet[t[i]-1])
                return ''.join(char_list)
        else:
            # process in batch mode
            assert t.numel() == length.sum(), f'length and len(text) should be equal. Got {t.numel()} for len(text) and {length.sum()} for length'
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[index:index+l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

# refers to https://github.com/meijieru/crnn.pytorch/blob/master/utils.py#L92
class Averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print(f'Size {str(v.size())}, Type: {v.data.type()}')
    print(f'| Max: {v.max().data[0]} | Min: {v.min().data[0]} | Mean: {v.mean.data[0]}')


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


