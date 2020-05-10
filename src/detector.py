import json
import os
import sys
import time
import zipfile
from collections import OrderedDict
from functools import cmp_to_key
from pathlib import Path

from PIL import Image

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from skimage import io

from detection.CRAFT.utils import imgproc
from detection.CRAFT.utils.CRAFT import getDetBoxes, adjustResultCoordinates
from detection.CRAFT.model import CRAFT

DATASET = (Path(__file__).parent / '..' / 'models').resolve()

def _copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class CRAFTDetector:
    cuda = False
    canvas_size = 1280
    magnify_ratio = 1.5
    text_threshold = 0.7
    link_threshold = 0.4
    low_text_score = 0.4
    enable_polygon = False
    enable_refiner = False
    trained_model = str(DATASET / 'craft_mlt_25k.pth')
    refiner_model = str(DATASET / 'craft_refiner_CTW1500.pth')

    def load(self):
        self.net = CRAFT()
        if torch.cuda.is_available():
            self.cuda = True
            self.net.load_state_dict(_copyStateDict(torch.load(self.trained_model)))
        else:
            # added compatibility for running on Mac
            self.net.load_state_dict(_copyStateDict(torch.load(self.trained_model, map_location='cpu')))

        if self.cuda:
            self.net = self.net.cuda()
            self.net = nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        # use refiner here
        self.refinenet = None
        if self.enable_refiner:
            from detector.modules.refinenet import RefineNet
            self.refinenet = RefineNet()
            if self.cuda:
                self.refinenet.load_state_dict(_copyStateDict(torch.load(self.refiner_model)))
                self.refinenet = self.refinenet.cuda()
                self.refinenet = nn.DataParallel(self.refinenet)
            else:
                self.refinenet.load_state_dict(_copyStateDict(torch.load(self.refiner_model, map_location='cpu')))

            self.refinenet.eval()
            self.enable_polygon = True

    def process(self, image):
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.magnify_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1) # [h x w x c] -> [c x h x w]
        x = torch.Tensor(x.unsqueeze(0))
        if self.cuda:
            x = x.cuda()
        y, feature = self.net(x)

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        if self.refinenet is not None:
            y_refiner = self.refinenet(y, feature)
            socre_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        boxes, polys = getDetBoxes(score_text,
                                   score_link,
                                   self.text_threshold,
                                   self.link_threshold,
                                   self.low_text_score,
                                   self.enable_polygon)
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
                return 1    # completely on below
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

        roi = list() # extract ROI
        for rect in sorted(rects, key=cmp_to_key(compare_rects)):
            x0, y0, x1, y1 = rect
            sub = image[x0:x1, y0:y1, :]
            roi.append(sub)

        return roi, boxes, polys, image
