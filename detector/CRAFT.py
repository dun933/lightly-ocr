"""UNet with vgg16"""

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import VGG16

from modules.conv import upconv, upsample

def CRAFT(input_tensor, backbone_weight=None, pooling=None, num_class=2, name='CRAFT', **kwargs):
    vgg = VGG16(include_top=False, input_tensor=input_tensor, weights=backbone_weight, pooling=pooling)
    # UNet
    upconv1 = upconv([512, 256])
    upconv2 = upconv([256, 128])
    upconv3 = upconv([128, 64])
    upconv4 = upconv([64, 32])

    # final layers for affinity region and scores
    cls = Sequential([
        Conv2D(32, kernel_size=3, padding='same'), Activation('relu'),
        Conv2D(32, kernel_size=3, padding='same'), Activation('relu'),
        Conv2D(16, kernel_size=3, padding='same'), Activation('relu'),
        Conv2D(16, kernel_size=1, padding='same'), Activation('relu'),
        Conv2D(32, kernel_size=num_class, padding='same'), Activation('sigmoid')
    ])

    slice5 = vgg.get_layer('block5_conv3').output
    x = MaxPooling2D(3, strides=1, padding='same', name='block5_pool')(slice5)
    x = Conv2D(1024, kernel_size=3, padding='same', dilation_rate=6)(x)
    x = Conv2D(1024, kernel_size=1)(x)

    # UNet start here
    x = upsample(target_layer=slice5)(x)
    x = Concatenate()([x, slice5])
    x = upconv1(x)

    slice4 = vgg.get_layer('block4_conv3').output
    x = upsample(target_layer=slice4)(x)
    x = Concatenate()([x, slice4])
    x = upconv2(x)

    slice3 = vgg.get_layer('block3_conv3').output
    x = upsample(target_layer=slice3)(x)
    x = Concatenate()([x, slice3])
    x = upconv3(x)

    slice2 = vgg.get_layer('block2_conv2').output
    x = upsample(target_layer=slice2)(x)
    x = Concatenate()([x, slice2])
    x = upconv4(x)

    # final layers
    x = cls(x)

    # returns region and affinity score
    return Lambda(lambda l: l[:,:,:,0])(x), Lambda(lambda l: l[:,:,:,1])(x)
