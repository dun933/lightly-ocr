import torch
from transform import TPS_STN
from backbone import ResNet
from sequence import Attention, biLSTM

class CRNN(nn.Module):
    # TPS - ResNet - biLSTM - Attn/CTC
    def __init__(self)
