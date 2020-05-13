import os
import io
import numpy

import torch
from torch import nn

import torch.utils.model_zoo as model_zoo
import torch.onnx as tonnx

from detection.CRAFT.model import CRAFT
from recognition.MORAN.model import MORAN
from recognizer import MORANRecognizer

model_parent = '../models'

def torch2onnx(model, parent_dir=model_parent, input_shape=(1, 3, 768, 768), cuda=False):
    # inputs [1,3,768,768] -> dummy default for CRAFT
    if cuda:
        if model == 'CRAFT':
            model = CRAFT()
        elif model == 'MORAN':
            model = MORAN(1, len(MORANRecognizer.alphabet.split(':')), 256, 32, 100, bidirectional=True, CUDA=True)
        model.load()
        model = model.net
        inputs = torch.randn(input_shape, device='cuda')
        tonnx.export(model.module, inputs, os.path.join(parent_dir, 'craft.onnx'), export_params=True, do_constant_folding=True, verbose=True,
                     input_names=['input'], output_names=['output'])
