from tensorflow.keras import layers


class FlowEstimator(layers.Layer):
    def __init__(self):
        super(FlowEstimator, self).__init__()
