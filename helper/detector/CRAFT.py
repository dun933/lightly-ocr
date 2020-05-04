"""
implements CRAFT backbone: UNet VGG16
- utilize [UNet](https://towardsdatascience.com/u-net-b229b32b4a71) architecture for segmentation
Reference:
    - [clovaai's CRAFT-pytorch codebase](https://github.com/clovaai/CRAFT-pytorch)
    - [Character Region Awareness for Text Detection](https://arxiv.org/pdf/1904.01941.pdf)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import VGG16
from modules.utils import upconv, convcls, upsamplelike

def VGG16UNet(weights=None, input_tensor=None, pooling=None):
    vgg16 = VGG16(include_top=False, weights=weights, input_tensor=input_tensor, pooling=pooling)
    block5 = vgg16.get_layer('block5_conv3').output
    x = MaxPooling2D(3,strides=1, padding='same', name='block5_pool1')(block5)
    x = Conv2D(1024, kernel_size=3, padding='same', dilation_rate=6)(x)
    x = Conv2D(1024, kernel_size=1)(x)

    x = upsamplelike(target_layer=block5, name='upsamplelike_1')(x)
    x = Concatenate()([x, block5])
    x = upconv(x, [512,256])

    block4 = vgg16.get_layer('block4_conv3').output
    x = upsamplelike(target_layer=block4, name='upsamplelike_2')(x)
    x = Concatenate()([x, block4])
    x = upconv(x,[256,128])

    block3 = vgg16.get_layer('block3_conv3').output
    x = upsamplelike(target_layer=block3, name='upsamplelike_3')(x)
    x = Concatenate()([x, block3])
    x = upconv(x, [128,64])

    block2 = vgg16.get_layer('block2_conv2').output
    x = upsamplelike(target_layer=block2, name='upsamplelike_4')(x)
    x = Concatenate()([x, block2])
    feature = upconv(x,[64,32])

    x = convcls(feature, 2)

    region_score = Lambda(lambda x: x[:,:,:,0])(x)
    affinity_score = Lambda(lambda x: x[:,:,:,1])(x)

    model = Model(input_tensor, [region_score, affinity_score], name='vgg16_unet')
    return model
