import tensorflow as tf
from tensorflow.keras.layers import *
import pickle

def load_data(pkl_path):
    with open(pkl_path,'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def upconv(input_tensor, filters):
    x = Conv2D(filters[0], kernel_size=1)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters[1], kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convcls(input_tensor, num_classes):
    '''
    args:
        - input_tensor<tf.Tensor>: 4D tensor
    return:
        - 4D tensor : region score and affinity score
    '''
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(input_tensor)
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(16, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv2D(16, kernel_size=num_classes, padding='same', activation='sigmoid')(x)
    return x

class upsamplelike(Layer):
    def __init__(self, target_layer, data_format='channel_last', **kwargs):
        super(upsamplelike, self).__init__(**kwargs)
        self.target_layer = target_layer
        self.target_shape = _collect_input_shape(target_layer)
        self.data_format=data_format
        self.inputspec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channel_last':
            h, w = self.target_shape[1], self.target_shape[2]
            return (input_shape[0], h, w, input_shape[3])
        else:
            raise ValueError(f'only support `channel_last`. got {self.data_format} instead')

    def call(self, inputs):
        return _resize_image(inputs, self.target_layer, self.target_shape, self.data_format)

# helper functions
def _collect_input_shape(input_tensors):
    '''return output shape of a list of tensors'''
    def _to_list(x):
        '''normalize a tensor to list'''
        if isinstance(x, list):
            return x
        return [x]
    input_tensors = _to_list(input_tensors)
    shapes = []
    for x in input_tensors:
        try:
            shapes.append(tf.keras.backend.int_shape(x))
        except Exception as e:
            print(e)
            shapes.append(None)
    if len(shapes) == 1:
        return shapes[0]
    return shapes

def _permute_dim(x, pattern):
    '''permute a tensor, wrapper for tf.transpose'''
    return tf.transpose(x, perm=pattern)

def _resize_image(x, target_layer, target_shape, data_format='channel_last'):
    '''resize image contained in a 4D tensor. set data_format to `channel_last`'''
    if data_format == 'channel_last':
        new_shape = tf.shape(target_layer)[1:3]
        x = tf.image.resize(x, new_shape, method='nearest')
        x.set_shape((None, target_shape[1], target_shape[2], None))
        return x
    else:
        raise ValueError(f'Unknown data_format: {str(data_format)}')
