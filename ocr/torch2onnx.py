import os
import sys

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.onnx
import yaml

from net import CRAFT, CRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(os.path.join('config.yml'), 'r') as f:
    config = yaml.safe_load(f)
os.makedirs(config['onnx_path'], exist_ok=True)
det, rec = config['pipeline'].split('-')
FUNCTION = {'CRAFT': CRAFT, 'CRNN': CRNN}


def torch2onnx(model_name, target_root=config['onnx_path'], debug=False):
    print(f'processing {model_name}')
    batch_size = 1
    lname = tuple()
    if not os.path.exists(os.path.join(target_root, f'{model_name}.onnx')):
        assert model_name in ['CRAFT', 'CRNN', 'MORANv2'], f'supports CRAFT, CRNN, MORANv2, got {model_name} instead, check `config.yml`'
        model = FUNCTION[f'{model_name}']()
        model.load()
        ocr = model.net
        converter = model.converter
        num_channels = 3 if model_name == 'CRAFT' else 1
        # dummy inputs
        dx = torch.randn(batch_size, num_channels, 244, 244, requires_grad=True).to(device)
        if model_name == 'CRAFT':
            outputs = ocr(dx)
        elif model_name == 'CRNN':
            text_pred = torch.LongTensor(batch_size, config['batch_max_len'] + 1).fill_(0).to(device)
            lname = (dx, text_pred)
            if config['prediction'] == 'Attention':
                outputs = ocr(dx, text_pred, training=False)
            else:
                outputs = ocr(dx, text_pred)
        else:
            max_iter = 20
            text = torch.LongTensor(1 * 5)
            length = torch.IntTensor(1)
            t, l = converter.encode('0' * max_iter)
            text.resize_(t.size()).copy_(t)
            length.resize_(l.size()).copy_(l)
            lname = (dx, length, text, text)
            outputs = ocr(dx, length, text, text, test=True, debug=True)
        # yapf: disable
        torch.onnx.export(ocr, lname, os.path.join(target_root, f'{model_name}.onnx'), export_params=True, verbose=True,
                          opset_version=11, do_constant_folding=True, input_names=['inputs'], output_names=['outputs'])
        # yapf: enable
        if debug:
            onnx_model = onnx.load(os.path.join(target_root), f'{model_name}.onnx')
            onnx.checker.check_model(onnx_model)
            onnx.helper.printable_graph(onnx_model)
    else:
        print(f'{model_name}.onnx already exists, continue.')


def inference(impath, onnx_path, mode='detect'):
    sess = onnxruntime.InferenceSession(onnx_path)
    image = cv2.imread(impath)


def to_numpy(tensor):
    try:
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
    except:
        return tensor.cpu().numpy()


# traced.save('converted_models/CRNN.pt')

# if __name__=='__main__':
#     for k, _ in FUNCTION.items():
#         torch2onnx(k, debug=True)
