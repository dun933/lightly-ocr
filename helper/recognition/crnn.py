"""
implements CRNN models as text recognition
"""
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

def crnn(input_tensor=None, input_shape=None, hidden_size=256, num_classes=1000, backbone='efficientnet',weights='imagenet', training=False, **kwargs):
    if not isinstance(input_tensor, tf.Tensor):
        if not isinstance(input_shape, tuple):
            raise AttributeError(f'if not provide an input tensor then need input_shape<tuple>, got {type(input_shape)} instead')
        else:
            inputs = Input(shape=input_shape, name='inputs')
    else:
        inputs = input_tensor
    # first check for cnn backbone
    if 'efficientnet' in backbone.lower():
        back_out = efficientnetB3(input_tensor=inputs, num_classes=num_classes,weights=weights).output
    elif 'resnet' in backbone.lower():
        back_out = resnet50v2(input_tensor=inputs, num_classes=num_classes,weights=weights).output
    else:
        raise KeyError(f'only support resnet and efficientnet. got {backbone} instead')
    # reshape layer from cnn to rnn
    inner = Reshape(target_shape=(int(back_out.shape[1]), int(back_out.shape[2] * back_out.shape[3])), name=f'reshape_connect')(back_out)
    inner = Dense(hidden_size // 4, activation='relu', kernel_initializer='he_normal', name=f'dense_connect')(inner)
    # rnn layer
    y_pred = biLSTM(inputs=inner, hidden_size=hidden_size, num_classes=num_classes)
    labels = Input(name='label', shape=[16,], dtype='float32')
    input_length = Input(name='input_length', shape=[1,], dtype='int64')
    label_length = Input(name='label_length', shape=[1,], dtype='int64')
    outputs = Lambda(ctc_lambda_fn, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=outputs, name=f'train_crnn_{backbone}')
    else:
        return Model(inputs=inputs, outputs=y_pred, name=f'pred_crnn_{backbone}')
