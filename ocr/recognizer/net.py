from collections import OrderedDict
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.autograd import Variable

from MORAN import dataset, tools
from MORAN.model import MORAN

MODEL_PATH = (Path(__file__).parent / 'models').resolve()


class MORANRecognizer:
    model_path = str(MODEL_PATH / 'moran_v2.pth')
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
            self.moran = self.moran.cuda()
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

        self.converter = tools.StringLabelConverter(self.alphabet, ':')
        self.transformer = MODEL_PATH.resizeNormalize((100, 32))

    def process(self, cv_img):
        image = Image.fromarray(cv_img).convert('L')
        image = self.transformer(image)
        if self.cuda:
            image = image.cuda()

        image = image.view(1, *image.size())
        image = Variable(image)
        text = torch.LongTensor(1 * 5)
        length = torch.IntTensor(1)
        text = Variable(text)
        length = Variable(length)

        t, l = self.converter.encode('0' * self.max_iter)
        tools.load_data(text, t)
        tools.load_data(length, l)
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
