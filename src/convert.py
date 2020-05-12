import os
import io
import numpy

from torch import nn

import torch.utils.model_zoo as model_zoo
import torch.onnx

from detector import CRAFTDetector
from recognizer import MORANRecognizer

model_parent = '../models'

def get_name(func):
    return func.net.module.__class__.__name__

def convert_torch(func, parent_dir, cuda=False):
    model = func().load()
