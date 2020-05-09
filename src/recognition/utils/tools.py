import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

class StringLabelConverter(object):
    '''convert between string and label for attention layer
    notes: insert `EOS` to alphabet for attention'''

    def __init__(self, alphabet, sep):
        '''
        args:
            - alphabet<str>: set of possible characters
            - ignore_case<bool>: whether or not to ignore all the case, default=True
        '''
        self._scanned_list = False
        self._out_of_list = ''
        self._ignore_case = True
        self.sep = sep
        self.alphabet = alphabet.split(sep)
        self.dict = {}
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

    def scan(self, text):
        tmp = text
        text = []
        for i in range(len(tmp)):
            res = ''
            for j in range(len(tmp[i])):
                char = tmp[i][j].lower() if self._ignore_case else tmp[i][j]
                if char not in self.alphabet:
                    if char in self._out_of_list:
                        continue
                    else:
                        self._out_of_list += char
                        file_out_of_list = open('ool.txt', 'a+')
                        file_out_of_list.write(char + '\n')
                        file_out_of_list.close()
                        print(f'{char}is not in alphabet...')
                        continue
                else:
                    res += char
            text.append(res)
        res = tuple(text)
        self._scanned_list = True
        return res

    def encode(self, text, scanned=True):
        '''
        support batch or single string
        args:
            - text<str or list>: text to convert
        return:
            - torch.IntTensor [length_0+length_1+...+length_(n-1)]: encoded text
            - torch.IntTensor [n]: len of each text
        '''
        self._scanned_list = scanned
        if not self._scanned_list:
            text = self.scan(text)

        if isinstance(text, str):
            text = [self.dict[char.lower() if self._ignore_case else char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length):
        '''decode encoded text back to string
        reverse encode()
        throws AssertionError: when text and length doesn\'t match'''
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, f'text with length {t.numel()} does not match with declared length {length}'
            return ''.join([self.alphabet[i] for i in t])
        else:
            # batch mode
            assert t.numel() == length.sum(), f'text with length {t.numel()} does not match with declared length {length}'
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                text.append(self.decode(t[index:index + l], torch.LongTensor([l])))
            index += l
        return texts

class Averager(object):
    '''compute average for torch.Tensor` (deprecated torch.Variable)'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()
        self.n_count += count
        self.sum += v

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def load_data(v, data):
    major, _ = get_torch_version()
    if major >= 1:
        v.resize_(data.size()).copy_(data)
    else:
        v.data.resize_(data.size()).copy_(data)

def get_torch_version():
    torch_version = str(torch.__version__).split('.')
    return int(torch_version[0]), int(torch_version[1])
