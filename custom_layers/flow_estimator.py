from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers


class FlowEstimator(layers.Layer):
    def __init__(self):
        super(FlowEstimator, self).__init__()

    @staticmethod
    def call(inputs):
        x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(inputs)
        x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=96, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)

        features = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        flow = Conv2D(filters=2, kernel_size=3, strides=1, padding='same')(features)
        return features, flow
