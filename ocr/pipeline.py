import argparse
import os
import time
from collections import OrderedDict

import cv2
import torch
import yaml

from net import CRAFT, CRNN, MORAN

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yml'), 'r') as f:
    CONFIG = yaml.safe_load(f)


def rename(key, rename='module'):
    if rename in key.split('.'):
        return key[len(rename) + 1:]
    else:
        return key


def rename_state_dict(source, fn=rename, target=None):
    # target is the new path to save source, default: write back to source
    # fn: function to transfer new key
    if target is None:
        target = source

    state_dict = torch.load(source)
    res = OrderedDict()

    for k, v in state_dict.items():
        nk = fn(k)
        res[nk] = v

    torch.save(res, target)


def calc_time(fn=lambda x, *a, **kw: x(*a, **kw)):
    start = time.time()
    fn()
    return f'{fn} took {time.time()-start:.3f}s'


# begin pipeline
def prepare(var, config=CONFIG):
    use_detector, use_recognizer = config['pipeline'].split('-')
    if use_detector == 'CRAFT':
        detector = CRAFT()
    else:
        raise AssertionError(f'only supported CRAFT atm. got {use_detector} instead')
    if use_recognizer == 'CRNN':
        recognizer = CRNN()
    elif use_recognizer == 'MORAN':
        recognizer = MORAN()
    else:
        raise AssertionError(f'only supports either CRNN or MORAN. got {use_recognizer} instead')
    mpaths = [detector.model_path, recognizer.model_path]
    for p in mpaths:
        rename_state_dict(p)
    # load from pretrained
    detector.load()
    recognizer.load()
    # prepare images
    image = var.img
    image = cv2.imread(image)  # use cv2 to read image
    return image, detector, recognizer


def getText(image, detector, recognizer):
    res = []
    use_recognizer = CONFIG['pipeline'].split('-')[1]
    with torch.no_grad():
        # detection
        roi, _, _, _ = detector.process(image)

        # recognition
        for _, img in enumerate(roi):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if use_recognizer == 'CRNN':
                text = recognizer.process(gray)
            elif use_recognizer == 'MORAN':
                text, _, _ = recognizer.process(gray)
            else:
                raise ValueError(f'using either CRNN or MORAN, got {use_recognizer} instead')
                break
            res.append(text)

    with open(os.path.join(os.path.dirname(os.path.relpath(__file__)), 'test', 'results.txt'), 'w') as f:
        for i in res:
            f.write(f'prediction: {i}\n')
        print(f'wrote results to {f.name}')
        f.close()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', help='path to config.yml, default is the same dir as pipeline')
    parser.add_argument('--img', required=True, help='image path for running ocr on')
    parser.add_argument('--debug', action='store_true', help='whether to run debug')
    var = parser.parse_args()

    image, detector, recognizer = prepare(var)
    getText(image, detector, recognizer)
    if var.debug:
        calc_time(prepare)
        calc_time(getText)
