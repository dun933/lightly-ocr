import os
from abc import ABC
from collections import OrderedDict
from functools import cmp_to_key

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from model import CRNNet, MORANet, VGG_UNet
from tools import dataset, imgproc
from tools.det_utils import adjustResultCoordinates, getDetBoxes
from tools.recog_utils import AttnLabelConverter, CTCLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(os.path.relpath(__file__)), 'models')
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yml'), 'r') as yf:
    config = yaml.safe_load(yf)


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class Detector(ABC):
    def load(self):
        pass

    def process(self, image):
        pass


class Recognizer(ABC):
    def load(self):
        pass

    def process(self, image):
        pass


class CRAFT(Detector):
    model_path = os.path.join(MODEL_PATH, 'CRAFT.pth')
    cuda = False
    canvas_size = 1280
    magnify_ratio = 1.5
    text_threshold = 0.7
    link_threshold = 0.4
    low_text_score = 0.4
    enable_polygon = False

    def load(self):
        self.net = VGG_UNet().to(device)
        if torch.cuda.is_available():
            self.cuda = True
            self.net.load_state_dict(copy_state_dict(torch.load(self.model_path)))
        else:
            # added compatibility for running on non-cuda device
            self.net.load_state_dict(copy_state_dict(torch.load(self.model_path, map_location='cpu')))

        if self.cuda:
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net).to(device)
            cudnn.benchmark = False
        self.net.eval()

    def process(self, image):

        with torch.no_grad():
            img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image,
                                                                                  self.canvas_size,
                                                                                  interpolation=cv2.INTER_LINEAR,
                                                                                  mag_ratio=self.magnify_ratio)
            ratio_h = ratio_w = 1 / target_ratio

            x = imgproc.normalizeMeanVariance(img_resized)
            x = torch.from_numpy(x).permute(2, 0, 1)  # [h x w x c] -> [c x h x w]
            x = torch.Tensor(x.unsqueeze(0)).to(device)
            y, feature = self.net(x)

            score_text = y[0, :, :, 0].cpu().data.numpy()
            score_link = y[0, :, :, 1].cpu().data.numpy()

            boxes, polys = getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text_score, self.enable_polygon)
            boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
            polys = adjustResultCoordinates(boxes, ratio_w, ratio_h)
            for k in range(len(polys)):
                if polys[k] is None:
                    polys[k] = boxes[k]

            rects = list()
            for box in boxes:
                poly = np.array(box).astype(np.int32)
                y0, x0 = np.min(poly, axis=0)
                y1, x1 = np.max(poly, axis=0)
                rects.append([x0, y0, x1, y1])

            def compare_rects(first_rect, second_rect):
                fx, fy, fxi, fyi = first_rect
                sx, sy, sxi, syi = second_rect
                if fxi <= sx:
                    return -1  # completely on above
                elif sxi <= fx:
                    return 1  # completely on below
                elif fyi <= fy:
                    return -1  # completely on left
                elif sxi <= sx:
                    return 1  # completely on right
                elif fy != sy:
                    return -1 if fy < sy else 1  # starts on more left
                elif fx != sx:
                    return -1 if fx < sx else 1  # top most when starts equally
                elif fyi != syi:
                    return -1 if fyi < syi else 1  # have least width
                elif fxi != sxi:
                    return -1 if fxi < sxi else 1  # have laast height
                else:
                    return 0  # same

            roi = list()  # extract ROI
            for rect in sorted(rects, key=cmp_to_key(compare_rects)):
                x0, y0, x1, y1 = rect
                sub = image[x0:x1, y0:y1, :]
                roi.append(sub)

        return roi, boxes, polys, image


class CRNN(Recognizer):
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    model_path = os.path.join(MODEL_PATH, 'CRNN.pth')
    net = CRNNet(config).to(device)
    config = config
    cuda = False
    converter = None
    transformer = None

    def load(self):
        if torch.cuda.is_available():
            self.cuda = True
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net).to(device)

        # print(f'loading pretrained from {self.model_path}')
        if self.cuda:
            self.net.load_state_dict(torch.load(self.model_path))
        else:
            self.net.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        if self.config['prediction'] == 'CTC':
            self.converter = CTCLabelConverter(self.alphabet)
        else:
            self.converter = AttnLabelConverter(self.alphabet)
        self.config['num_classes'] = len(self.converter.character)
        self.transformer = dataset.resize_normalize((100, 32))

        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

    def process(self, image):
        batch_size = 1  # since we only process one image
        image = Image.fromarray(image).convert('L')
        # print(image.size)
        image = self.transformer(image).to(device)
        image = image.view(1, *image.size()).to(device)
        len_pred = torch.IntTensor([self.config['batch_max_len'] * batch_size]).to(device)
        text_pred = torch.LongTensor(batch_size, self.config['batch_max_len'] + 1).fill_(0).to(device)

        if self.config['prediction'] == 'CTC':
            preds = self.net(image, text=text_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_idx = preds.max(2)
            preds_idx = preds_idx.view(-1)
            raw_pred = self.converter.decode(preds_idx.data, preds_size.data)
        else:
            preds = self.net(image, text=text_pred, training=False)
            _, preds_idx = preds.max(2)
            raw_pred = self.converter.decode(preds_idx, len_pred)

        probs = F.softmax(preds, dim=2)
        max_probs, _ = probs.max(dim=2)
        with open(os.path.join(os.path.dirname(os.path.relpath(__file__)), 'test', 'log_results.txt'), 'w+') as f:
            for max_prob in max_probs:
                # returns prediction here
                if self.config['prediction'] == 'Attention':
                    try:
                        # [EOS] token process in a list
                        pred_EOS = raw_pred[0].index('[s]')
                        raw_pred = raw_pred[0][:pred_EOS]
                        max_prob = max_prob[:pred_EOS]
                    except ValueError:
                        print('Not found EOS token, continue.\n(potential error)')
                        continue  # when there isn't a EOS token
                confidence = max_prob.cumprod(dim=0)[-1]
                f.write(f'results: {raw_pred}\t\tconfidence score: {confidence:.4f}\n')
                print(f'results: {raw_pred}\t\tconfidence score: {confidence:.4f}')
        return raw_pred


class MORAN(Recognizer):
    model_path = os.path.join(MODEL_PATH, 'MORANv2.pth')
    alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'
    max_iter = 20
    cuda = False
    net = None
    state_dict = None
    converter = None
    transformer = None

    def load(self):
        if torch.cuda.is_available():
            self.cuda = True
            self.net = MORANet(1, len(self.alphabet.split(':')), 256, 32, 100, bidirectional=True)
            self.net = self.net.to(device)
        else:
            self.net = MORANet(1, len(self.alphabet.split(':')), 256, 32, 100, bidirectional=True, input_data_type='torch.FloatTensor')
        if self.cuda:
            self.state_dict = torch.load(self.model_path)
        else:
            self.state_dict = torch.load(self.model_path, map_location='cpu')

        MORAN_state_dict_rename = OrderedDict()
        for k, v in self.state_dict.items():
            name = k.replace('module.', '')
            MORAN_state_dict_rename[name] = v
        self.net.load_state_dict(MORAN_state_dict_rename)

        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

        self.converter = AttnLabelConverter(self.alphabet, sep=':')
        self.transformer = dataset.resize_normalize((100, 32))

    def process(self, image):

        with torch.no_grad():
            image = Image.fromarray(image).convert('L')
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
            dataset.load_data(text, t)
            dataset.load_data(length, l)
            output = self.net(image, length, text, text, test=True, debug=True)

            preds, preds_rev = output[0]
            out_img = output[1]

            _, preds = preds.max(1)
            _, preds_rev = preds_rev.max(1)

            sim_preds = self.converter.decode(preds.data, length.data)
            sim_preds = sim_preds.strip().split('$')[0]
            sim_preds_rev = self.converter.decode(preds_rev.data, length.data)
            sim_preds_rev = sim_preds_rev.strip().split('$')[0]

        return sim_preds, sim_preds_rev, out_img
