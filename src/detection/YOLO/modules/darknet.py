from tensorflow.keras.layers import *
from tensorflow.keras import Model
from .layers import conv2d_unit, resblock

def stack_resblock(input_tensor, filters, num_blocks, training=False):
    x = resblock(input_tensor, filters, training=training)

    for _ in range(num_blocks - 1):
        x = resblock(x, filters, training=training)
    return x

def darknet53(input_tensor, name='darknet53', training=False, **kwargs):
    x = conv2d_unit(input_tensor, filters=32, kernel_size=3, training=training)

    x = conv2d_unit(x, 64, 3, strides=2, training=training)
    x = stack_resblock(x, 32, num_blocks=1, training=training)

    x = conv2d_unit(x, 128, 3, strides=2, training=training)
    x = stack_resblock(x, 64, num_blocks=2, training=training)

    x = conv2d_unit(x, 256, 3, strides=2, training=training)
    x_36 = x = stack_resblock(x, 128, num_blocks=8, training=training)

    x = conv2d_unit(x, 512, 3, strides=2, training=training)
    x_61 = x = stack_resblock(x, 256, num_blocks=8, training=training)

    x = conv2d_unit(x, 1024, 3, strides=2, training=training)
    x = stack_resblock(x, 512, num_blocks=4, training=training)

    return Model(inputs=input_tensor, outputs=[x, x_61, x_36], name=name)
