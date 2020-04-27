"""
Transform network layer --> uses to transform image as inputs for CNN. ported from clovaai's pytorch implementation
# References
    [clovaai](https://arxiv.org/pdf/1904.01906.pdf) --> https://github.com/clovaai/deep-text-recognition-benchmark
    [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)
    [Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/pdf/1603.03915.pdf)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *

class tps_stn(Model):
    '''rectification Network of RARE, or TPS based STN'''

    def __init__(self, F, I_size, I_r_size, I_channels=1):
        '''
        args:
            - F<int32>: fiducial points for grid generator
            - I_size<tuple, (h,w)>: (height, width) of the input image I
            - I_r_size<tuple, (h,w)>: (height, width) of the rectified image I_r
            - I_channels<int32>: channels of input layers
        returns:
            - batch_I_r<tf.Tensor>: rectified image [batch x I_r_height x I_r_width x I_channels]
        '''
        super(tps_stn, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_channels = I_channels
        self.LocalizationNetwork = LocalizationNet(self.F, self.I_channels)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def call(self, inputs):
        batch_C_prime = self.LocalizationNetwork(inputs)  # batch x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime) # batch x n (=I_r_width x I_r_height) x 2
        build_P_prime_reshape = tf.reshape(build_P_prime, [build_P_prime.shape[0], self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = bilinear_sampler(inputs, build_P_prime_reshape)
        return batch_I_r

class LocalizationNet(Layer):
    """Localization Network of RARE, which predicts C' (Kx2) from I (I_width x I_height)"""

    def __init__(self, F, I_channels):
        '''
        args:
            - batch_I<tf.Tensor>: input_image [batch_size x I_height x I_width X I_channels]
            - F<int32>: fiducial points for grid generator
            - I_channels<int32>: channels of input layers
        returns:
            - batch_C_prime<tf.Tensor>: predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        '''
        super(LocalizationNet, self).__init__()
        self.F = F
        self.I_channels = I_channels
        self.cnn = Sequential(
            [Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False), BatchNormalization(), Activation('relu'),
             MaxPool2D(pool_size=(2, 2)), # [batch_size x I_height/2 x I_width/2 X 64]
             Conv2D(128, 3, 1, padding='same', use_bias=False), BatchNormalization(), Activation('relu'),
             MaxPool2D(pool_size=(2, 2)), # [batch_size x I_height/4 x I_width/4 x 128]
             Conv2D(256, 3, 1, padding='same', use_bias=False), BatchNormalization(), Activation('relu'),
             MaxPool2D(pool_size=(2, 2)), # [batch_size x I_height/8 x I_width/8 x 256]
             Conv2D(512, 3, 1, padding='same', use_bias=False), BatchNormalization(), Activation('relu'),
             GlobalAveragePooling2D() # [batch_size x 512]
             ], name='localnet')
        self.localization_fc1 = Sequential([Dense(256), Activation('relu')], name='local_fc_1')
        self.localization_fc2 = Dense(self.F * 2, use_bias=True, kernel_initializer='zeros', bias_initializer=self._fc2_bias(self.F), name='local_fc_2')

    def _fc2_bias(self, F):
        # init fc2 a bit different
        # see RARE paper Fig. 6 (a)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = tf.concat([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return tf.reshape(initial_bias, [2,int(F/2), 2])

    def call(self, batch_I):
        batch_size = batch_I.shape[0]
        feats = tf.reshape(self.cnn(batch_I), (batch_size, -1))
        batch_C_prime = tf.reshape(self.localization_fc2(self.localization_fc1(feats)), (batch_size, self.F, 2))
        return batch_C_prime


class GridGenerator(Layer):
    """Grid Generator of RARE, producing P_prime by T*P"""

    def __init__(self, F, I_r_size):
        '''
        Generate P_hat and inv_delta_C for later
        args:
            - F <int32>: number of fiducial points for the grid generator
            - I_r_size<list>: resize image size [h, w]
        '''
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size[0], I_r_size[1]
        self.F = F
        self.C = self._build_C(self.F)
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        # for fine tuning with different width, don't have to register buffer
        self.inv_delta_C = tf.Tensor(self._build_inv_delta_C(self.F, self.C), dtype='float32')
        self.P_hat = tf.Tensor(self._build_P_hat(self.F, self.C, self.P), dtype='float32')

    def _build_C(self, F):
        '''returns coordinates of fiducial points in I_r, C'''
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        '''returns inv_delta_C which is needed to calculate T'''
        hat_C = np.zeros((F, F), dtype='float')
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1), # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1), # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1) # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        '''Generate Grid from batch_C_prime [batch_size x F x 2]'''
        batch_size = batch_C_prime.shape[0]
        batch_inv_delta_C = tf.repeat(self.inv_delta_C, repeats=[batch_size, 1, 1])
        batch_P_hat = tf.repeat(self.P_hat, repeats=[batch_size, 1, 1])
        batch_C_primes_with_zeros = tf.concat([batch_C_prime, tf.zeros(shape=[batch_size, 3, 2], dtype='tf.float32')], axis=1)
        batch_T = tf.matmul(batch_inv_delta_C, batch_C_primes_with_zeros)
        batch_P_prime = tf.matmul(batch_P_hat, batch_T)
        return batch_P_prime

def bilinear_sampler(img, grid):
    """
    args:
        - img<tf.Tensor>: batch of images in (B, H, W, C) layout.
        - grid<tf.Tensor>: output of grid generator
    returns:
        - out: interpolated images according to grids. Same size as grid.
    """
    def get_pixel_value(img, x, y):
        """
        args:
                - img: tensor of shape (B, H, W, C)
                - x: flattened tensor of shape (B*H*W,)
                - y: flattened tensor of shape (B*H*W,)
        returns:
                - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, [batch_size, 1, 1])
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    x, y = grid[..., 0], grid[..., 1]
    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out
