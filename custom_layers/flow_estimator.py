import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D


class FlowEstimator(layers.Layer):
    def __init__(self):
        super(FlowEstimator, self).__init__()
        k_init = tf.keras.initializers.he_normal()
        self.conv_1 = Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer=k_init)
        self.conv_2 = Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer=k_init)
        self.conv_3 = Conv2D(filters=96, kernel_size=3, padding='same', kernel_initializer=k_init)
        self.conv_4 = Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer=k_init)
        self.features = Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer=k_init)
        self.flow = Conv2D(filters=2, kernel_size=3, padding='same')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_3(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_4(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.features(x)
        features = tf.nn.leaky_relu(x, alpha=0.1)
        flow = self.flow(features)
        return features, flow
