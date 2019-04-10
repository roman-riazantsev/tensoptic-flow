from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers


class FeatureExtractor(layers.Layer):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    @staticmethod
    def call(inputs):
        lvls = []
        x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
        x = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        lvls.append(x)

        x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        lvls.append(x)

        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        lvls.append(x)

        x = Conv2D(filters=96, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=96, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        lvls.append(x)

        x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        lvls.append(x)

        x = Conv2D(filters=196, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = Conv2D(filters=196, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(filters=196, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        lvls.append(x)
        return lvls
