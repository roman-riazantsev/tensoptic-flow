import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers


class FeatureExtractor(layers.Layer):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        k_init = tf.keras.initializers.he_normal()

        self.conv_1_a = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', kernel_initializer=k_init)
        self.conv_1_b = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)
        self.conv_1_c = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)

        self.conv_2_a = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', kernel_initializer=k_init)
        self.conv_2_b = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)
        self.conv_2_c = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)

        self.conv_3_a = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer=k_init)
        self.conv_3_b = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)
        self.conv_3_c = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)

        self.conv_4_a = Conv2D(filters=96, kernel_size=3, strides=2, padding='same', kernel_initializer=k_init)
        self.conv_4_b = Conv2D(filters=96, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)
        self.conv_4_c = Conv2D(filters=96, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)

        self.conv_5_a = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer=k_init)
        self.conv_5_b = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)
        self.conv_5_c = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)

        self.conv_6_a = Conv2D(filters=196, kernel_size=3, strides=2, padding='same', kernel_initializer=k_init)
        self.conv_6_b = Conv2D(filters=196, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)
        self.conv_6_c = Conv2D(filters=196, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init)

    def call(self, inputs):
        lvls = []

        x = self.conv_1_a(inputs)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_1_b(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_1_c(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        lvls.append(x)

        x = self.conv_2_a(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_2_b(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_2_c(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        lvls.append(x)

        x = self.conv_3_a(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_3_b(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_3_c(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        lvls.append(x)

        x = self.conv_4_a(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_4_b(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_4_c(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        lvls.append(x)

        x = self.conv_5_a(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_5_b(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_5_c(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        lvls.append(x)

        x = self.conv_6_a(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_6_b(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv_6_c(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        lvls.append(x)

        return lvls
