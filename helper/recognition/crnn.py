"""
implements CRNN models as text recognition
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
# from efficientnet import efficientnetB3
# from resnet50 import resnet50v2
# from biLSTM import biLSTM

def ctc_lambda_fn(args):
    y_pred, labels, input_length, label_length=args
    y_pred  = y_pred[:,2:,:]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def crnn(backbone='efficientnet', input_tensor=None, input_shape=None, hidden_size=128, outsize=64, num_classes=1000, training=False, **kwargs):
    if not isinstance(input_tensor, tf.Tensor):
        if not isinstance(input_shape, tuple):
            raise AttributeError(f'if not provide an input tensor then need input_shape<tuple>, got {type(input_shape)} instead')
        else:
            inputs = Input(shape=input_shape, name='inputs')
    else:
        inputs = input_tensor
    # first check for backbone
    if backbone=='efficientnet':
        back_out = efficientnet(input_tensor=inputs).output
    elif backbone=='resnet':
        back_out - resnet50v2(input_tensor=inputs).output
    else:
        raise KeyError(f'only support resnet and efficientnet. got {backbone} instead')

    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=outputs, name=f'train_crnn_{backbone}')
    else:
        return Model(inputs=inputs, outputs=outputs, name=f'pred_crnn_{backbone}')
