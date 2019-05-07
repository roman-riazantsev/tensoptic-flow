from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras import layers
import tensorflow as tf


class UpsamplingLayer(layers.Layer):
    def __init__(self, upsampling_type):
        super(UpsamplingLayer, self).__init__()
        self.upsampling_type = upsampling_type
        self.conv_2d_t = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same')
        self.up_2d = UpSampling2D()
        self.conv_2d = Conv2D(filters=2, kernel_size=3, padding='same')

    def call(self, x):
        if self.upsampling_type == 'Conv2DTranspose':
            x = self.conv_2d_t(x)
        elif self.upsampling_type == 'UpSampling2D':
            x = self.up_2d(x)
            x = self.conv_2d(x)
        return x
