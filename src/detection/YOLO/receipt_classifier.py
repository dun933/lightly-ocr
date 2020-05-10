import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

from .modules.crf import CRF
from .modules.layers import conv2d_unit, CBAM

class BiLSTMClassifier(Model):
    def __init__(self, hidden_size, num_classes, name='rnn-classifier', **kwargs):
        super(BiLSTMClassifier, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.cell = [
            LSTMCell(self.hidden_size, kernel_regularizer=l2(), recurrent_regularizer=l2(), dropout=0.2, recurrent_dropout=0.2),
            LSTMCell(self.hidden_size, kernel_regularizer=l2(), recurrent_regularizer=l2(), dropout=0.2, recurrent_dropout=0.2)
        ]
        self.rnn1 = Bidirectional(RNN(self.cell, return_sequences=True))
        self.rnn2 = Bidirectional(LSTM(num_classes, return_sequences=True, activation='softmax', kernel_regularizer=l2(),
                                       recurrent_regularizer=l2(), dropout=0.2, recurrent_dropout=0.2), merge_mode='sum')
        self.crf = CRF(num_classes)

    def call(self, inputs, training=None, training_embedding=None, mask=None):
        x = self.rnn1(inputs)
        x = self.rnn2(x)
        x = self.crf(x)

        return x

def ASPP(input_tensor, filters, grid_size, training=False, name='aspp'):
    features = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    features = conv2d_unit(features, filters=filters, kernel_size=1, training=training)
    features = UpSampling2D(grid_size)(features)

    dilated_conv1 = conv2d_unit(input_tensor, filters, kernel_size=1, training=training)
    dilated_conv4 = conv2d_unit(input_tensor, filters, kernel_size=(3, 5), dilation_rate=4, training=training)
    dilated_conv8 = conv2d_unit(input_tensor, filters, kernel_size=(3, 5), dilation_rate=8, training=training)
    dilated_conv16 = conv2d_unit(input_tensor, filters, kernel_size=(3, 5), dilation_rate=16, training=training)

    x = Concatenate()([features, dilated_conv1, dilated_conv4, dilated_conv8, dilated_conv16])
    x = conv2d_unit(x, filters=filters, kernel_size=1, training=training)
    return x

def CBAM_resnet(input_tensor, num_classes, grid_size, training=None, training_embedding=None, mask=None, name='CBAM-ResNet'):
    x = conv2d_unit(input_tensor, filters=64, kernel_size=(3, 5))

    # CBAM-ResNet
    short_1 = x
    x = conv2d_unit(x, filters=256, kernel_size=(3, 5), training=training)
    x = CBAM(x, filters=256, reduction=16, training=training)
    x = Concatenate()([x, short_1])

    short_2 = x
    x = conv2d_unit(x, filters=256, kernel_size=(3, 5), training=training)
    x = CBAM(x, filters=256, reduction=16, training=training)
    x = Concatenate()([x, short_2])

    # dilated convblock
    x = conv2d_unit(x, filters=256, kernel_size=(3, 5), dilation_rate=2, training=training)
    x = conv2d_unit(x, filters=256, kernel_size=(3, 5), dilation_rate=4, training=training)
    x = conv2d_unit(x, filters=256, kernel_size=(3, 5), dilation_rate=8, training=training)
    x = conv2d_unit(x, filters=256, kernel_size=(3, 5), dilation_rate=16, training=training)

    x = ASPP(x, filters=256,grid_size=grid_size, training=training)
    x = Concatenate()([x, short_1])
    x = conv2d_unit(x, filters=64, kernel_size=1, training=training)
    x = conv2d_unit(x, num_classes, 1, activation=False, bn=False)

    return Model(inputs=input_tensor, outputs=x, name=name)
