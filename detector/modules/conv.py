import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
import numpy as np


def tensor2list(x):
    return x if isinstance(x, list) else [x]

def collect_input_shape(input_tensors):
    input_tensors = tensor2list(input_tensors)
    shapes = []
    for x in input_tensors:
        try:
            shapes.append(x.shape)
        except Exception as e:
            print(e)
            shapes.append(None)
    if len(shapes)==1:
        return shapes[0]
    return shapes

def permute_dims(input_tensor, pattern):
    return tf.transpose(input_tensor, perm=pattern)

def resize_img(input_tensor, target_layer, target_shape, data_format):
    if data_format =='channels_first':
        new = tf.shape(target_layer)[2:]
        x = permute_dims(input_tensor, [0,2,3,1])
        x = tf.image.resize(x, new, method='nearest')
        x = permute_dims(x, [0,3,1,2])
        x.set_shape((None, None, target_shape[2], target_shape[3]))
        return x
    elif data_format == 'channels_last':
        new = tf.shape(target_layer)[1:3]
        x = tf.image.resize(input_tensor, new, method='nearest')
        x.set_shape((None, target_shape[1], target_shape[2], None))
        return x
    else:
        raise ValueError(f'uknown data_format: {data_format}')

class upconv(Layer):
    def __init__(self, filters):
        super(upconv, self).__init__()
        self.filters = filters
        self.conv = Sequential([
            Conv2D(self.filters[0], kernel_size=1), BatchNormalization(), Activation('relu'),
            Conv2D(self.filters[1],kernel_size=3,padding='same'),BatchNormalization(), Activation('relu')
            ])

    def call(self, inputs):
        return self.conv(inputs)

class upsample(Layer):
    def __init__(self, target_layer, data_format='channels_last', **kwargs):
        super(upsample, self).__init__(**kwargs)
        self.target_layer = target_layer
        self.target_shape = collect_input_shape(target_layer)
        self.data_format = data_format
        self.inputspec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1],self.target_shape[2], self.target_shape[3])
        elif self.data_format == 'channels_last':
            return(input_shape[0], self.target_shape[1], self.target_shape[2], input_shape[3])

    def call(self, inputs, **kwargs):
        return resize_img(inputs, self.target_layer, self.target_shape, self.data_format)
