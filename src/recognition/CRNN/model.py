import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import *

class CRNN(Model):
    def __init__(self, num_classes, training):
        super(CRNN, self).__init__()

        kernel_initializer = tf.random_normal_initializer(0,0.05)
        bias_initializer = tf.constant_initializer(value=0.)

        self.conv1 = Conv2D(filters=64, kernel_size=3, padding='same',
                            activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.pool1 = MaxPool2D(pool_size=(2,2),strides=2)

        self.conv2 = Conv2D(filters=64, kernel_size=3, padding='same',
                            activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.pool2 = MaxPool2D(pool_size=(2,2),strides=2)

        self.conv3 = Conv2D(filters=256, kernel_size=3, padding='same',
                            activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.bn3 = BatchNormalization(trainable=training)

        self.conv4 = Conv2D(filters=256, kernel_size=3, padding='same',
                            activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.pool4 = MaxPool2D(pool_size=(2,2),strides=(2,1))

        self.conv5 = Conv2D(filters=512, kernel_size=3, padding='same',
                            activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.bn5 = BatchNormalization(trainable=training)

        self.conv6 = Conv2D(filters=512, kernel_size=3, padding='same',
                            activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.pool6 = MaxPool2D(pool_size=(2,2),strides=(2,1))

        self.conv7 = Conv2D(filters=512, kernel_size=2, padding='valid',
                            activation='relu',kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

        self.lstm_fw_cell_1 = LSTM(256, return_sequences=True)
        self.lstm_bw_cell_1 = LSTM(256,go_backwards=True, return_sequences=True)
        self.birnn1 = Bidirectional(layer=self.lstm_fw_cell_1, backward_layer=self.lstm_bw_cell_1)

        self.lstm_fw_cell_2 = LSTM(256, return_sequences=True)
        self.lstm_bw_cell_2 = LSTM(256,go_backwards=True, return_sequences=True)
        self.birnn2 = Bidirectional(layer=self.lstm_fw_cell_2, backward_layer=self.lstm_bw_cell_2)

        self.dense = Dense(num_classes, activation='relu', kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.pool1(x)

        x = self.conv2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x, training=training)
        x = self.bn3(tf.cast(x, dtype=tf.float32),training=training)

        x = self.conv4(x, training=training)
        x = self.pool4(x)

        x = self.conv5(x, training=training)
        x = self.bn5(tf.cast(x, dtype=tf.float32),training=training)

        x = self.conv6(x, training=training)
        x = self.pool6(x)

        x = self.conv7(x, training=training)

        x = tf.reshape(x, [-1, x.shape[2],x.shape[3]]) # -> [b x time x filters]

        x = self.birnn1(x)
        x = self.birnn2(x)

        logits = self.dense(x)

        pred = tf.argmax(tf.nn.softmax(logits), axis=2)

        outputs = tf.transpose(logits, [1,0,2])

        return logits, pred, outputs
