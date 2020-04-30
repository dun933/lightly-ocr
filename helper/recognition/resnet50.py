"""
modified ResNet V2 implementation in Keras
    - ported from keras-applications
    - Using LeakyReLU for faster training time
Reference:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
    - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)
    - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) (CVPR 2017)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from functools import wraps

# test wrapper functions of resnet50v2
@wraps(ResNet50V2)
def resnet50_wrapper(inputs, **kwargs):
    '''
    wrapper of ResNet50V2, include_top=False
    args:
        inputs<tf.Tensor>: 4d tensors with [batch, h, w, c]
        include_top<bool>: whether added top layers ,returns True or False
    returns:
        resnet50v2
    reference:
        https://github.com/tensorflow/tensorflow/tensorflow/python/keras/applications/resnet_v2.py#L33
        [Indentity Mappings in Deep Residual Network](https://arxiv.org/pdf/1603.05027.pdf)'''
    kwrapper = {'weights': None}
    kwrapper['input_tensor'] = inputs if isinstance(inputs, tf.Tensor) else None
    # only need input_shape when include_top=False
    kwrapper['input_shape'] = inputs.shape[1:] if not kwargs.get('include_top') else None
    kwrapper.update(kwargs)
    return ResNet50V2(**kwrapper)

#############################################################################################################
# channel_last so axis=3
# ResNet with LeakyReLU to fix dying ReLU
# ported from keras-applicaions/resnet-common.py

def block_2(x, filters, kernel_size=3, strides=1, axis=3, shortcut=False, name=None):
    '''
    residual block, implements preactivation to weights
    args:
        - x<tf.Tensor>: inputs tensor
        - filters<int32>: filters for layer
        - kernel_size<int32, default=3>: kernel size of layer
        - strides<int32, default=1>: strides of layer
        - shortcut<boolean, default=False>: implements residual shortcut
        - name<str>: name of the block
    returns:
        - outputs<tf.Tensor>: output tensor from the block'''
    preact = BatchNormalization(axis=axis, epsilon=1.001e-5, name=f'{name}_preact_bn')(x)
    preact = LeakyReLU(name=f'{name}_preact_leaky')(preact)
    if shortcut:
        shortcut = Conv2D(4 * filters, 1, strides=strides, name=f'{name}_0_conv')(preact)
    else:
        shortcut = MaxPool2D(1, strides=strides)(x) if strides > 1 else x
    x = Conv2D(filters, 1, strides=1, use_bias=False, name=f'{name}_1_conv')(preact)
    x = BatchNormalization(axis=axis, epsilon=1.001e-5, name=f'{name}_1_bn')(x)
    x = LeakyReLU(name=f'{name}_1_leaky')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=f'{name}_2_pad')(x)
    x = Conv2D(filters, kernel_size, strides=strides, use_bias=False, name=f'{name}_2_conv')(x)
    x = BatchNormalization(axis=axis, epsilon=1.001e-5, name=f'{name}_2_bn')(x)
    x = LeakyReLU(name=f'{name}_2_leaky')(x)
    x = Conv2D(4 * filters, 1, name=f'{name}_3_conv')(x)
    x = Add(name=f'{name}_out')([shortcut, x])
    return x

def stack_2(x, filters, blocks, stride1=2, name=None):
    '''
    stacked residual blocks
    args:
        - x<tf.Tensor>: input tensor
        - filters<int32>: filters of bot layer
        - blocks<int32>: # of blocks in the network
        - stride1<int32, default=2>: stride of the last layer in the stack
        - name<str>: name of the stacked
    returns:
        - outputs<tf.Tensor>: output tensor for the second stack'''
    x = block_2(x, filters, shortcut=True, name=f'{name}_block1')
    for i in range(2, blocks):
        x = block_2(x, filters, name=f'{name}_block_{str(i)}')
    x = block_2(x, filters, strides=stride1, name=f'{name}_block_{str(blocks)}')
    return x

def resnet(stack_fn, preact=True, use_bias=True, model_name='resnet', axis=3, input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    '''
    full resnet blown resnet backbone
    args:
        - stack_fn<functions>:function returns output tensor for stacked residual blocks
        - preact<bool, default=True>: use preactivation, true for resnet_v2
        - use_bias<bool, default=True>: use biases for convolution layer, true for resnet_v2, resnet
        - axis<int32, default=3>: axis for batch normalization, default is 3 since we are dealing with channel_last only
        - model_name<str>: give it a cool name
        - input_tensor<tf.Tensor>: output of Input() as image inputs of model
        - input_shape<tuple optional>: shape of input, optional
        - pooling<str from ['None','avg','max'], default='None'>: optional pooling for feature extraction, default is None to get 4D tensor output of last convolution layer
        - classes<int, default=1000>: number of classes to classify images, depends on the dataset
    return:
        - resnet<tf.Model>: a model instance'''
    # TODO: added pooling options for FC layers
    if not isinstance(input_tensor, tf.Tensor):
        if not isinstance(input_shape, tuple):
            raise AttributeError(f'if not provide an input tensor then need input_shape<tuple>, got {type(input_shape)} instead')
        else:
            inputs = Input(shape=input_shape, name='inputs')
    else:
        inputs = input_tensor
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
    x = Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
    if preact is False:
        x = BatchNormalization(axis=axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPool2D(3, strides=2, name='pool1_pool')(x)
    x = stack_fn(x)
    if preact is True:
        x = BatchNormalization(axis=axis, epsilon=1.001e-5, name='post_bn')(x)
        x = Activation('relu', name='post_relu')(x)
    return Model(input_tensor, x, name=model_name)

# resnet50v2
def resnet50v2(input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    def stack_fn(x):
        x = stack_2(x, 64, 3, name='conv2')
        x = stack_2(x, 128, 4, name='conv3')
        x = stack_2(x, 256, 6, name='conv4')
        x = stack_2(x, 512, 3, stride1=1, name='conv5')
        return x
    model = resnet(stack_fn, preact=True, use_bias=True, model_name='resnet50v2', axis=3,
                   input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes, **kwargs)
    return model
