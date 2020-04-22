import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from functools import wraps

# features extractor from input image, tested between darknet53 and resnet50(wrapper around the function from keras)

# resnet50
@wraps(ResNet50V2)
def resnet(inputs, **kwargs):
    # https://github.com/tensorflow/tensorflow/blob/0ec3c78e0a34f224ee041b54a24717f77d3246fa/tensorflow/python/keras/applications/resnet.py#L61
    # https://github.com/tensorflow/tensorflow/blob/f31621823a2467daf7cf40e9e2d3a56ce193e695/tensorflow/python/keras/applications/resnet_v2.py#L33
    '''kwrapper of ResNet50V2, include_top=False
        args:
            inputs<tf.Tensor>: 4d tensors with [batch, h, w, c]
            include_top<bool>: whether added top layers ,returns True or False
        returns:
            resnet50v2
        Reference: [Indentity Mappings in Deep Residual Network](https://arxiv.org/pdf/1603.05027.pdf)'''
    kwrapper = {'weights': None}
    kwrapper['input_tensor'] = inputs if isinstance(inputs, tf.Tensor) else None
    # only need input_shape when include_top=False
    kwrapper['input_shape'] = inputs.shape[1:] if not kwargs.get('include_top') else None
    kwrapper.update(kwargs)
    return ResNet50V2(**kwrapper)


def test(debug=True):
    inputs = Input(shape=(320, 320, 3), name='inputs')
    resnet50 = resnet(inputs, include_top=False)
    if debug:
        resnet50.summary()


# test()
