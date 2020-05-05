"""
implements bi-directional LSTM as a sequence labeling
Reference:
    - [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Add, BatchNormalization, Concatenate, Dense, Activation

def biLSTM(inputs, hidden_size, characters='', **kwargs):
    '''
    args:
        inputs<tf.Tensor>: connecting layers shape [batch, timesteps, features]
        hidden_size<int32>: hidden dims
    returns:
        output(contextual features)
        3d tf.Tensor shape [batch, timesteps, features]
    '''
    def stack_func(inp):
        args = [hidden_size]
        fkwargs = {'return_sequences': True, 'kernel_initializer': 'he_normal'}
        bkwargs = {}
        bkwargs.update(fkwargs)
        bkwargs['activation'] = 'relu'
        bkwargs['go_backwards'] = True
        return [LSTM(*args, **fkwargs)(inp), LSTM(*args, **bkwargs)(inp)]
    # inner = Reshape(target_shape=((int(inputs.shape[1]), int(inputs.shape[2] * inputs.shape[3]))), name='reshaped_inner')(inputs)
    # inner = Dense(hidden_size // 4, activation='relu', name='fc_1')(inner)
    # after fixing intermediate layers
    merged_1 = Add()(stack_func(inputs))
    merged_1 = BatchNormalization()(merged_1)
    merged_2 = Concatenate()(stack_func(merged_1))
    merged_2 = BatchNormalization()(merged_2)
    outputs = Dense(len(characters), kernel_initializer='he_normal', activation='softmax', name='dense_b4_ctc')(merged_2)
    # since we only care about the output rather than the model itself, return y_pred instead of Model(inputs, outputs)
    return outputs
