"""
Sequence Labeling layer. The output from CNN then feeds toward biLSTM/biGRU for sequence labeling. Output feeds to Joint CTC-Attention
# Reference
    - [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
# biLSTM/biGRU


def call_func(func, *args, **kwargs):
    return func(*args, **kwargs)


def bisequence(inputs, hidden_size, output_size, num_classes=1000, cell='LSTM', debug=False, **kwargs):
    '''implements bi-directional LSTM
        args:
            inputs<tf.Tensor>: connecting layers shape [batch, timesteps, features]
            hidden_size<int32>: hidden dims
            output_size<int32>: output dims
        returns:
            output(contextual features)
            3d tf.Tensor shape [batch, timesteps, features]
            '''
    def stack_fn(inp, cell):
        args = [hidden_size]
        fkwargs = {'return_sequences': True, 'kernel_initializer': 'he_normal'}
        bkwargs = {}
        bkwargs.update(fkwargs)
        bkwargs['activation'] = 'relu'
        bkwargs['go_backwards'] = True
        if cell.lower() == 'lstm':
            return [call_func(LSTM, *args, **fkwargs)(inp), call_func(LSTM, *args, **bkwargs)(inp)]
        elif cell.lower() == 'gru':
            return [call_func(GRU, *args, **fkwargs)(inp), call_func(GRU, *args, **bkwargs)(inp)]
        else:
            raise ValueError(f'only supported LSTM/GRU, got {cell} instead')
    # started bisequence
    inner = Reshape(target_shape=((int(inputs.shape[1]), int(inputs.shape[2] * inputs.shape[3]))), name='reshaped_inner')(inputs)
    inner = Dense(output_size // 4, activation='relu', name='fc_1')(inner)
    # after fixing intermediate layers
    merged_1 = Add()(stack_fn(inner, cell))
    merged_1 = BatchNormalization()(merged_1)
    merged_2 = Concatenate()(stack_fn(merged_1, cell))
    merged_2 = BatchNormalization()(merged_2)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name='finals')(merged_2)
    model = Model(inputs, outputs, name=f'bi{cell}')
    if debug:
        model.compile(optimizer='adam', loss='mse')
        model.summary()
    return model

# just for testing
# bisequence(Input(shape=(10,10,2048), name='inputs'), 256,64,debug=True)
