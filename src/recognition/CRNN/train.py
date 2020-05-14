import os, sys, time, random, string

import yaml
import torch
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data

from utils import *
from dataset import hierarchical_dataset, align_collate, BalancedDataset
from model import CRNN

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yml'), 'r') as f:
    config = yaml.safe_load(f)



