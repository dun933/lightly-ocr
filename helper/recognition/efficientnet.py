"""
EfficientNet B3 implementation in Keras
- ported from keras-applications
# Reference paper
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
"""
import os
import math
import tensorflow as tf
from copy import deepcopy
from tensorflow.keras import Model
from tensorflow.keras.layers import *

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling. Use `channel_last`
    args:
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    returns:
        A tuple.
    """
    img_dim = 1
    input_size = inputs.shape[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def swish(x):
    return x * tf.nn.sigmoid(x)

def block(inputs, activation_fn=swish, drop_rate=0., name='', filters_in=32, filters_out=16, kernel_size=3, strides=1, expand_ratio=1, se_ratio=0., id_skip=True, axis=3):
    '''inverted residual block.
    args:
        inputs<tf.Tensor>: input tensor
        activation_fn: activation function
        drop_rate<int32>: float between 0 and 1 fir dropout rate
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
        axis: batch location for `channel_last`
    returns:
        output tensor for the block.
    '''
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = Conv2D(filters, 1, padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_expand_conv')(inputs)
        x = BatchNormalization(axis=axis, name=f'{name}_expand_bn')(x)
        x = Activation(activation_fn, name=f'{name}_expand_swish')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = ZeroPadding2D(padding=correct_pad(x, kernel_size),
                          name=f'{name}_dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = DepthwiseConv2D(kernel_size,
                        strides=strides,
                        padding=conv_pad,
                        use_bias=False,
                        depthwise_initializer=CONV_KERNEL_INITIALIZER,
                        name=f'{name}_dwconv')(x)
    x = BatchNormalization(axis=axis, name=f'{name}_bn')(x)
    x = Activation(activation_fn, name=f'{name}_activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = GlobalAveragePooling2D(name=f'{name}_se_squeeze')(x)
        if axis == 1:
            se = Reshape((filters, 1, 1), name=f'{name}_se_reshape')(se)
        else:
            se = Reshape((1, 1, filters), name=f'{name}_se_reshape')(se)
        se = Conv2D(filters_se, 1,
                    padding='same',
                    activation=activation_fn,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name=f'{name}_se_reduce')(se)
        se = Conv2D(filters, 1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name=f'{name}_se_expand')(se)
        x = multiply([x, se], name=f'{name}_se_excite')

    # Output phase
    x = Conv2D(filters_out, 1,
               padding='same',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=f'{name}_project_conv')(x)
    x = BatchNormalization(axis=axis, name=f'{name}_project_bn')(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = Dropout(drop_rate,
                        noise_shape=(None, 1, 1, 1),
                        name=f'{name}_drop')(x)
        x = add([x, inputs], name=f'{name}_add')

    return x

def efficientnet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 input_tensor=None,
                 input_shape=None,
                 axis=3,
                 activation_fn=swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 classes=1000,**kwargs):
    if not isinstance(input_tensor, tf.Tensor):
        if not isinstance(input_shape, tuple):
            raise AttributeError(f'if not provide an input tensor then need input_shape<tuple>, got {type(input_shape)} instead')
        else:
            inputs = Input(shape=input_shape, name='inputs')
    else:
        inputs = input_tensor

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # build stem
    x = inputs
    x = ZeroPadding2D(padding=correct_pad(x, 3),name='stem_conv_pad')(x)
    x = Conv2D(round_filters(32), 3,strides=2, padding='valid', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name='stem_conv')(x)
    x = BatchNormalization(axis=axis, name='stem_bn')(x)
    x = Activation(activation_fn, name='stem_swish')(x)

    # build blocks
    block_args = deepcopy(blocks_args)
    b = 0
    blocks = float(sum(args['repeats'] for args in block_args))
    for (i, args) in enumerate(block_args):
        assert args['repeats'] > 0
        # update block inputs and output filters based on depth multiplier
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation_fn, drop_connect_rate * b / blocks,
                      name=f'block{i+1}{chr(j+97)}', **args)
            b += 1

    # Build top
    x = Conv2D(round_filters(1280), 1,
               padding='same',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name='top_conv')(x)
    x = BatchNormalization(axis=axis, name='top_bn')(x)
    x = Activation(activation_fn, name='top_activation')(x)
    return Model(inputs, x,name=model_name)

def efficientnetB3(input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   **kwargs):
    return efficientnet(1.2, 1.4, 300, 0.3,
                        model_name='efficientnet-b3',
                        input_tensor=input_tensor, input_shape=input_shape, classes=classes,
                        **kwargs)
