import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Flatten, Conv2D
from feature_extraction import resnet50v2
from sequence_labeling import bisequence
from transformation import spatialtransformer

def ctc_lambda_func(args):
    # from keras documentation
    iy_pred, ilabels, iinput_length, ilabel_length = args
    iy_pred = iy_pred[:, 2:, :]  # no such influence
    return tf.keras.backend.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)



def net(input_shape=(320, 320, 3), hidden=256, output_size=64, num_classes=1000, debug=False):
    inputs = Input(shape=input_shape, name='inputs')
    # STN started here
        # reshape to [batch, w, h*dims], already included in bisequence
    # feeds through resnet50v2
    resnetV2 = resnet50v2(input_tensor=inputs, classes=num_classes)
    res_out = resnetV2.output
    # Then feeds output to a spatial transformer network, still need some fixing
    # added biLSTM here, consider between LSTM and GRU
    lstm = bisequence(res_out, hidden, output_size, num_classes=num_classes,debug=debug)
    lstm_out = lstm.output
    infer = Model(inputs=inputs, outputs=out, name=f'infer_{input_shape[0]}')
    # added ctc_lost
    labels = Input(name='label', shape=[label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([out, labels, input_length, label_length])
    train = Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss], name=f'train_{input_shape[0]}')
    return train, infer

if __name__ == '__main__':
    train, infer = net()
    train.compile(loss={'ctc':lambda y_true, y_pred: y_pred}, optimizer='sgd')
    train.summary()
