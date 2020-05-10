import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

def conv2d_unit(input_tensor, filters, kernel_size, strides=1, dilation_rate=1, padding='same',
                activation=True, bn=True, training=False, **kwargs):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=tf.random_normal_initializer(0., 0.05), kernel_regularizer=l2())(input_tensor)
    if bn:
        x = BatchNormalization()(x, training=training)
    if activation:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def resblock(input_tensor, filters, training=False, **kwargs):
    # didn't include zero padding for residual connection
    x = conv2d_unit(input_tensor=input_tensor, filters=filters, kernel_size=(1, 1), training=training)
    x = conv2d_unit(input_tensor=x, filters=filters * 2, kernel_size=(3, 3), activation=False, training=training)
    x = Add()([input_tensor, x])
    x = LeakyReLU()(x)
    return x

def CBAM(input_tensor, filters, reduction, name='convblock-attn', training=False, **kwargs):
    # channel attention
    x_mean = tf.reduce_mean(input_tensor, axis=(1, 2), keepdims=True)
    x_mean = conv2d_unit(x_mean, filters // reduction, 1, bn=False, training=training)
    x_mean = conv2d_unit(x_mean, filters, 1, activation=False, bn=False, training=training)

    x_max = tf.reduce_max(input_tensor, axis=(1, 2), keepdims=True)
    x_max = conv2d_unit(x_max, filters // reduction, 1, bn=False, training=training)
    x_max = conv2d_unit(x_max, filters, 1, activation=False, bn=False, training=training)

    x = x_mean + x_max
    x = Activation(tf.nn.sigmoid)(x)
    x = tf.multiply(input_tensor, x)

    # spatial attention
    y_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    y_max = tf.reduce_max(x, axis=-1, keepdims=True)
    y = Concatenate()([y_mean, y_max])
    y = conv2d_unit(y, filters=1, kernel_size=7, bn=False, activation=False, training=training)
    y = Activation(tf.nn.sigmoid)(y)
    y = tf.multiply(x, y)

    return y
