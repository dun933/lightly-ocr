import collections
import string
import sys
import unittest

import torch

from tools import utils


def equal(a1,a2):
    if isinstance(a1, torch.Tensor):
        return a1.equal(a2)
    elif isinstance(a1, str):
        return a1==a2
    elif isinstance(a1, collections.Iterable):
        res = True
        for (i,j) in zip(a1,a2):
            res = res and equal(i,j)
        return res
    else:
        return a==b

def utils_test(unittest.TestCase):
    def check_converter(self):
        encoder = utils.CTCLabelConverter(string.ascii_lowercase)

        res = encoder.encode('fifa')
        tar = (torch.IntTensor([6,9,6, 1]), torch.IntTensor([4]))
        self.assertTrue(equal(res, tar))

        res = encoder.encode(['eff','abc'])
        tar = ( torch.IntTensor([5,6,6,1,2,3]), torch.IntTensor([3,3]) )
        self.assertTrue(equal(res, tar))
