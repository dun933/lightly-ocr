import os
import json
import random
import itertools

import numpy as np
from tqdm import tqdm

import utils


def cocotext_dataset(data_dir='data',split='train', train_data_dir='~/Downloads/train2014/',get_label=False):
    dataset = []
    coco_json = os.path.join(data_dir, 'cocotext.v2.json')
    with open(coco_json, 'r') as f:
        labels = json.loads(f.read())
    sel_ids = [ids for ids, data in labels['imgs'].items() if data['set'] in split]
    sel_fname = [labels['imgs'][cocoid]['file_name'] for cocoid in sel_ids]
    for sids in sel_ids:
        fpath = os.path.join(train_data_dir, sel_fname[sel_ids.index(sids)])
        for annotation_idx in labels['imgToAnns'][sel_ids]:
            ann = labels['anns'][str(annotation_idx)]
            dataset.append((fpath, np.array(ann['mask']).reshape(-1,2),ann['utf8-string']))
    if get_label:
        return dataset, (labels, train_data_dir)
    return dataset

def detector_imgen(labels, width, height,augmenter=None, area_threshold=0.5, focused=False, min_area=None):
    '''generate augmented (image, lines) from a list of (fpath, lines, confindence)
    args:
        - labels: (img, lines, confindence)
        - augmenter: augmenter used to apply to images
    '''
    labels = labels.copy()
    for idx in itertools.cycle(range(len(labels))):
        if idx == 0:
            random.shuffle(labels)
        im_fpath, lines, confindence = labels[idx]
        img = utils.read(im_fpath)
        if augmenter is not None:
            img, lines = utils.augment(boxes=lines, boxes_format='lines',
                                       image=img, area_threshold=area_threshold, min_area=min_area, augmenter=augmenter)
        img, scale = utils.fit(img, w=width, h=height, mode='letterbox', return_scale=True)
        lines = utils.adjust_boxes(boxes=lines, boxes_format='lines', scale=scale)
        yield img, lines, confindence

def recognizer_imgen(labels, height, width, alphabets, augmenter=None):
    '''gen augmented image for recognizer
    labels shape (fpath, box, labels)'''
    illegal = sum(any(c not in alphabets for c in text) for _,_, text in labels)
    if illegal>0:
        print(f'{illegal}/{len(labels)} insteances have illegal character')
    labels = labels.copy()
    for idx in itertools.cycle(range(len(labels))):
        if idx == 0:
            random.shuffle(labels)
        fpath, box, text = labels[idx]
        cval = np.random.randint(low=0, high=255, size=3).astype('uint8')
        if box is not None:
            img = utils.warpbox(image=utils.read(fpath),
                                box=box.astype('float32'),
                                target_height=height,
                                target_width=width,
                                cval=cval)
        else:
            img = utils.read_fit(fpath=fpath,width=width, height=height, cval=cval)
        text = ''.join([c for c in text if c in alphabets])
        if augmenter:
            img = augmenter.augment_image(img)
        yield (img, text)



