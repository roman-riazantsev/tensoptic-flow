from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers
import tensorflow as tf


class FlowRefiner(layers.Layer):
    def __init__(self):
        super(FlowRefiner, self).__init__()
        k_init = tf.keras.initializers.he_normal()
        self.conv_1 = Conv2D(filters=128, kernel_size=3, dilation_rate=1, padding='same', kernel_initializer=k_init)
        self.conv_2 = Conv2D(filters=128, kernel_size=3, dilation_rate=2, padding='same', kernel_initializer=k_init)
        self.conv_3 = Conv2D(filters=128, kernel_size=3, dilation_rate=4, padding='same', kernel_initializer=k_init)
        self.conv_4 = Conv2D(filters=96, kernel_size=3, dilation_rate=8, padding='same', kernel_initializer=k_init)
        self.conv_5 = Conv2D(filters=64, kernel_size=3, dilation_rate=16, padding='same', kernel_initializer=k_init)
        self.conv_6 = Conv2D(filters=32, kernel_size=3, dilation_rate=1, padding='same', kernel_initializer=k_init)
        self.conv_7 = Conv2D(filters=2, kernel_size=3, dilation_rate=1, padding='same', kernel_initializer=k_init)

    def call(self, features, flow):
        x = self.conv_1(features)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_3(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_4(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_5(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_6(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_7(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        refined_flow = flow + x
        return refined_flow