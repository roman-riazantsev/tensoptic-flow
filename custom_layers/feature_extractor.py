from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers
import tensorflow as tf


class FeatureExtractor(layers.Layer):
    def __init__(self, pyramid_levels):
        super(FeatureExtractor, self).__init__()

        self.pyramid_levels = pyramid_levels
        channel_numbers = [16, 32, 64, 96, 128, 196]
        act = lambda t: tf.nn.leaky_relu(t, alpha=0.1)
        k_init = tf.keras.initializers.he_normal()
        self.convs_a = []
        self.convs_b = []
        self.convs_c = []

        for i in range(self.pyramid_levels):
            self.convs_a.append(
                Conv2D(filters=channel_numbers[i], kernel_size=3, strides=2, padding='same', activation=act,
                       kernel_initializer=k_init))
            self.convs_b.append(
                Conv2D(filters=channel_numbers[i], kernel_size=3, strides=1, padding='same', activation=act,
                       kernel_initializer=k_init))
            self.convs_c.append(
                Conv2D(filters=channel_numbers[i], kernel_size=3, strides=1, padding='same', activation=act,
                       kernel_initializer=k_init))

    def call(self, inputs):
        lvls = []
        x = inputs
        for i in range(self.pyramid_levels):
            x = self.convs_a[i](x)
            x = self.convs_b[i](x)
            x = self.convs_c[i](x)
            lvls.append(x)

        return lvls
