import os
import itertools
import random

import cv2
import keras_ocr
import numpy as np

MJSYNTH = '/mnt/Vault/centralized/database/mjsynth/'

def mj_fpath(mjsynth, split):
    with open(os.path.join(MJSYNTH, f'annotation_{split}.txt'),'r') as f:
        fpath = [os.path.join(MJSYNTH, l.split(' ')[0][2:]) for l in f.readlines()]
    return fpath

def imgen(fpath, augmenter, width, height):
    fpath = fpath.copy()
    for f in itertools.cycle(fpath):
        txt = f.split(os.sep)[-1].split('_')[1].lower()
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = keras_ocr.tools.fit(img, width=width, height=height,
                                  cval=np.random.randint(low=0, high=255, size=3).astype('uint8'))
        if augmenter is not None:
            img = augmenter.augment_image(img)
        if f == fpath[-1]:
            random.shuffle(f)
        yield img, txt





