import os
from collections import OrderedDict
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from .CRNN.model import CRNN
from .CRNN.tools import dataset
from .CRNN.tools import utils as crnn_utils
from .MORAN import utils as moran_utils
from .MORAN.model import MORAN

# from torch.autograd import Variable

MODEL_PATH = (Path(__file__).parent / 'models' / 'pretrained').resolve()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MORANRecognizer:
    model_path = str(MODEL_PATH / 'MORANv2.pth')
    alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'
    max_iter = 20
    cuda = False
    moran = None
    state_dict = None
    converter = None
    transformer = None

    def load(self):
        if torch.cuda.is_available():
            self.cuda = True
            self.moran = MORAN(1,
                               len(self.alphabet.split(':')),
                               256,
                               32,
                               100,
                               bidirectional=True,
                               CUDA=True)
            self.moran = self.moran.to(device)
        else:
            self.moran = MORAN(1,
                               len(self.alphabet.split(':')),
                               256,
                               32,
                               100,
                               bidirectional=True,
                               inputDataType='torch.FloatTensor',
                               CUDA=False)

        print(f'loading pretrained model from {self.model_path}')
        if self.cuda:
            self.state_dict = torch.load(self.model_path)
        else:
            self.state_dict = torch.load(self.model_path, map_location='cpu')

        MORAN_state_dict_rename = OrderedDict()
        for k, v in self.state_dict.items():
            name = k.replace('module.', '')
            MORAN_state_dict_rename[name] = v
        self.moran.load_state_dict(MORAN_state_dict_rename)

        for p in self.moran.parameters():
            p.requires_grad = False
        self.moran.eval()

        self.converter = moran_utils.AttnLabelConverter(self.alphabet, ':')
        self.transformer = dataset.resize_normalize((100, 32))

    def process(self, cv_img):
        image = Image.fromarray(cv_img).convert('L')
        image = self.transformer(image)
        if self.cuda:
            image = image.cuda()

        image = image.view(1, *image.size())
        # image = Variable(image)
        text = torch.LongTensor(1 * 5)
        length = torch.IntTensor(1)
        # text = Variable(text)
        # length = Variable(length)

        t, l = self.converter.encode('0' * self.max_iter)
        moran_utils.load_data(text, t)
        moran_utils.load_data(length, l)
        output = self.moran(image, length, text, text, test=True, debug=True)

        preds, preds_rev = output[0]
        out_img = output[1]

        _, preds = preds.max(1)
        _, preds_rev = preds_rev.max(1)

        sim_preds = self.converter.decode(preds.data, length.data)
        sim_preds = sim_preds.strip().split('$')[0]
        sim_preds_rev = self.converter.decode(preds_rev.data, length.data)
        sim_preds_rev = sim_preds_rev.strip().split('$')[0]

        return sim_preds, sim_preds_rev, out_img


class CRNNRecognizer:
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    model_path = str(MODEL_PATH / 'CRNN.pth')
    with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CRNN',
                         'config.yml'), 'r') as f:
        config = yaml.safe_load(f)
    crnn = CRNN(config)
    cuda = False
    converter = None
    transformer = None

    def load(self):
        if torch.cuda.is_available():
            self.crnn = self.crnn.cuda()
            self.cuda = True

        print(f'loading pretrained from {self.model_path}')
        if self.cuda:
            self.crnn.load_state_dict(torch.load(self.model_path))
        else:
            self.crnn.load_state_dict(
                torch.load(self.model_path, map_location='cpu'))
        if self.config['prediction'] == 'CTC':
            self.converter = crnn_utils.CTCLabelConverter(self.alphabet)
        else:
            self.converter = crnn_utils.AttnLabelConverter(self.alphabet)
        self.transformer = dataset.resize_normalize((100, 32))

        for p in self.crnn.parameters():
            p.requires_grad = False
        self.crnn.eval()

    def process(self, cv_img):
        image = Image.open(cv_img).convert('L')
        batch_size = image.size(0)
        image = self.transformer(image)
        if self.cuda:
            image.image.cuda()
        image = image.view(1, *image.size()).to(device)
        len_pred = torch.IntTensor([self.config['batch_max_len'] * batch_size
                                    ]).to(device)
        text_pred = torch.LongTensor(batch_size, self.config['batch_max_len'] +
                                     1).fill_(0).to(device)

        if self.config['prediction'] == 'CTC':
            preds = self.crnn(image, text_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_idx = preds.max(2)
            preds_idx = preds_idx.view(-1)
            raw_pred = self.converter.decode(preds_idx.data, preds_size.data)
        else:
            preds = self.crnn(image, text_pred, training=False)
            _, preds_idx = preds.max(2)
            raw_pred = self.converter.decode(preds_idx, len_pred)

        probs = F.softmax(preds, dim=2)
        max_probs, _ = probs.max(dim=2)
        for max_prob in max_probs:
            # returns prediction here
            if self.config['prediction'] == 'Attention':
                pred_EOS = raw_pred.find('[s]')
                raw_pred = raw_pred[:pred_EOS]
                max_prob = max_prob[:pred_EOS]
            confidence = max_prob.cumprod(dim=0)[-1]
        return raw_pred, confidence
