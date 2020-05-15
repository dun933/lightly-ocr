import torch.nn as nn
from morn import MORN
from asrn_resnet import ASRN

class MORAN(nn.Module):
    def __init__(self, nc, nclass, nh, targetH, targetW, bidirectional=False, inputDataType='torch.cuda.FloatTensor',maxBatch=256,CUDA=True):
        super(MORAN, self).__init__()
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA)
        self.ASRN = ASRN(targetH, nc, nclass, nh, bidirectional, CUDA)

    def forward(self, x, length, text, text_rev, test=False, debug=False):
        if debug:
            x_rectified, img = self.MORN(x, test, debug=debug)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds, img
        else:
            x_rectified = self.MORN(x, test, debug=debug)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds
