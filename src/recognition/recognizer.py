"""
implements CRNN models as text recognition
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from modules.efficientnet import efficientnetB3
from modules.resnet50 import resnet50v2
from modules.biLSTM import biLSTM

def ctc_lambda_fn(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def crnn(input_tensor=None, input_shape=None, hidden_size=256, num_classes=1000, characters='', backbone='efficientnet', weights='imagenet', training=False, **kwargs):
    if not isinstance(input_tensor, tf.Tensor):
        if not isinstance(input_shape, tuple):
            raise AttributeError(f'if not provide an input tensor then need input_shape<tuple>, got {type(input_shape)} instead')
        else:
            inputs = Input(shape=input_shape, name='inputs')
    else:
        inputs = input_tensor
    # first check for cnn backbone
    if 'efficientnet' in backbone.lower():
        back_out = efficientnetB3(input_tensor=inputs, num_classes=num_classes, weights=weights).output
    elif 'resnet' in backbone.lower():
        back_out = resnet50v2(input_tensor=inputs, num_classes=num_classes, weights=weights).output
    else:
        raise KeyError(f'only support resnet and efficientnet. got {backbone} instead')
    # reshape layer from cnn to rnn
    inner = Reshape(target_shape=(int(back_out.shape[1]), int(back_out.shape[2] * back_out.shape[3])), name='reshape_connect')(back_out)
    inner = Dense(hidden_size, activation='relu', kernel_initializer='he_normal', name='dense_connect')(inner)
    # rnn layer
    y_pred = biLSTM(inputs=inner, hidden_size=hidden_size, characters=characters)
    labels = Input(name='label', shape=[y_pred.shape[1], ], dtype='float32')
    input_length = Input(name='input_length', shape=[1, ], dtype='int64')
    label_length = Input(name='label_length', shape=[1, ], dtype='int64')
    outputs = Lambda(ctc_lambda_fn, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=outputs, name=f'train_crnn_{backbone}')
    else:
        return Model(inputs=inputs, outputs=y_pred, name=f'pred_crnn_{backbone}')


class Recognizer:
    def __init__(self, input_tensor=None, input_shape=None, hidden_size=256, num_classes=1000, alphabets=None, backbone='efficientnet', weights=None):
        assert alphabets or weights, 'needed either weights or alphabets, got None instead'
        self.alphabets = alphabets
        self.weights = weights
        self.train_crnn_model = crnn(input_tensor=input_tensor, input_shape=input_shape, hidden_size=hidden_size, num_classes=num_classes,
                                     alphabets=self.alphabets, backbone=backbone, weights=self.weights, training=True)
        self.pred_crnn_model = crnn(input_tensor=input_tensor, input_shape=input_shape, hidden_size=hidden_size, num_classes=num_classes,
                                    alphabets=self.alphabets, backbone=backbone, weights=self.weights, training=False)

    def get_batch_generator(self, image_generator, batch_size=5, lowercase=False):
        y = np.zeros((batch_size, 1))
        len_max_str = self.train_crnn_model.input_shape[1][1]
        while True:
            batch = [sample for sample, _ in zip(image_generator, range(batch_size))]
            if not self.train_crnn_model.input_shape[-1] == 3:
                img = [cv2.cvtColor(sample[0], cv2.COLOR_BGR2GRAY)[..., np.newaxis] for sample in batch]
            else:
                img = [sample[0] for sample in batch]
            img = np.array([im.astype('float32') / 255 for im in img])
            sentence = [sample[1].strip() for sample in batch]
            if lowercase:
                sentence = [s.lower() for s in sentence]
            label_len = np.array([len(s) for s in sentence])[:, np.newaxis]
            labels = np.array([[self.alphabets.index(c) for c in s] + [-1] * (len_max_str - len(s)) for s in sentence])
            input_length = np.ones((batch_size, 1)) * len_max_str
            if len(batch[0]) == 3:
                sample = np.array([sample[2] for sample in batch])
                yield (img, labels, input_length, label_len), y, sample
            else:
                yield (img, labels, input_length, label_len), y
