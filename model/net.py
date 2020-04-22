import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Flatten, Conv2D
from feature_extraction import resnet
from sequence_labeling import bisequence
from transform_layer import spatialtransformer

def ctc_lambda_func(args):
    # from keras documentation
    iy_pred, ilabels, iinput_length, ilabel_length = args
    iy_pred = iy_pred[:, 2:, :]  # no such influence
    return tf.keras.backend.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)

def localnet(input_shape):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    w = np.zeros((64, 6), dtype='float32')
    weights = [w, b.flatten()]

    loc_input = Input(input_shape)

    loc_conv_1 = Conv2D(16, (5, 5), padding='same', activation='relu')(loc_input)
    loc_conv_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(loc_conv_1)
    loc_fla = Flatten()(loc_conv_2)
    loc_fc_1 = Dense(64, activation='relu')(loc_fla)
    loc_fc_2 = Dense(6, weights=weights)(loc_fc_1)

    output = Model(inputs=loc_input, outputs=loc_fc_2)

    return output

def net(input_shape=(320, 320, 3), hidden=256, output_size=64, num_classes=1000, label_len=16, training=True):
    inputs = Input(shape=input_shape)
    # feeds through resnet50
    resnet50 = resnet(inputs, include_top=False)
    outputs = resnet50.output
    shape = outputs.get_shape()
    # STN started here
    loc_inputs = (shape[1], shape[2], shape[3])
    stn = spatialtransformer(localization_net=localnet(loc_inputs), output_size=(loc_inputs[0], loc_inputs[1]))(outputs)
    # reshape to [batch, w, h*dims], already included in bisequence
    # added biLSTM here, consider between LSTM and GRU
    lstm = bisequence(stn, hidden, output_size, num_classes=num_classes, training=training)
    out = lstm.output
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
