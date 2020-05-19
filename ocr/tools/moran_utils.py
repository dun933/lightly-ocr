import collections

import torch
import torch.nn as nn


# also ported from utils.py in CRNN with some slight modification
class AttnLabelConverter(object):
    # convert between string and label for attention layer
    def __init__(self, alphabet, sep):
        self._scanned = False
        self._ool = ''
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
                    if char in self._ool:
                        continue
                    else:
                        self._ool += char
                        f_ool = open('out_of_list.txt', 'a+')
                        f_ool.write(char + '\n')
                        f_ool.close()
                        print(f'{char} is not in alphabet...')
                        continue
                else:
                    res += char
            text.append(res)
        res = tuple(text)
        self._scanned = True
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
        self._scanned = scanned
        if not self._scanned:
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
        throws AssertionError: when text and length does not match'''
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
                texts.append(self.decode(t[index:index + l], torch.LongTensor([l])))
            index += l
        return texts
