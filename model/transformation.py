"""
Transform network layer --> uses to transform image as inputs for CNN. ported from clovaai's pytorch implementation
# References
    [clovaai](https://arxiv.org/pdf/1904.01906.pdf) --> https://github.com/clovaai/deep-text-recognition-benchmark
    [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)
    [Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/pdf/1603.03915.pdf)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Activation, Dense, Reshape

class LocalizationNet(Layer):
    """Localization Network of RARE, which predicts C' (Kx2) from I (I_width x I_height)"""

    def __init__(self, F, I_channels):
        '''
        args:
            - F<int32>: fiducial points for gridgenerator
            - I_channels<int32>: channels of input layers
        returns:
            - instance of Localization Net
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
        self.localization_fc2 = LinearFC2(self.F * 2,self.F, input_dim=256)

    def call(self, batch_I):
        """
        args:
            - batch_I<tf.Tensor>: input_image [batch_size x I_height x I_width X I_channels]
        returns:
            - batch_C_prime<tf.Tensor>: predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.shape[0]
        feats = Reshape(target_shape=(batch_size, -1))(self.cnn(batch_I))
        batch_C_prime = Reshape(target_shape=(batch_size, self.F, 2))(self.localization_fc2(self.localization_fc1(feats)))
        return batch_C_prime

class LinearFC2(Layer):
    '''
    A tweak in dense layer to use custom initial_bias, init weight matrix with zeros for fc2 layer in RARE
    '''
    def __init__(self,units, F, input_dim=32):
        '''
        args:
            - units<int32>: dimensionality of output space
            - F<int32>: fiducial points
            - input_dim<int32, default=32>: just your input dimension
        output:
             - your densely tensor with shape [batch_size, units]
        '''
        super(LinearFC2, self).__init__()
        self.w = tf.Variable(initial_value=tf.zeros_initializer()(shape=(input_dim, units), dtype=tf.float32),trainable=True)
        self.b = self._fc2_bias(F)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)+self.b

    def _fc2_bias(self, F):
        # init fc2 a bit different
        # see RARE paper Fig. 6 (a)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return tf.convert_to_tensor(initial_bias, dtype=tf.float32)

x = tf.ones((2,2))
l = LinearFC2(64,20,256)
print(l(x))

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
        '''Generate GRid from batch_C_prime [batch_size x F x 2]'''
        batch_size = batch_C_prime.shape[0]
        batch_inv_delta_C = tf.repeat(self.inv_delta_C, repeats=[batch_size, 1, 1])
        batch_P_hat = tf.repeat(self.P_hat, repeats=[batch_size, 1, 1])
        batch_C_primes_with_zeros = tf.concat([batch_C_prime, tf.zeros(shape=[batch_size, 3, 2], dtype='tf.float32')], axis=1)
        batch_T = tf.matmul(batch_inv_delta_C, batch_C_primes_with_zeros)
        batch_P_prime = tf.matmul(batch_P_hat, batch_T)
        return batch_P_prime
