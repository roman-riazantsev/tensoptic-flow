import tensorflow as tf
from tensorflow.python.keras.layers import concatenate

from custom_layers.cost_volume_layer import CostVolumeLayer
from custom_layers.flow_estimator import FlowEstimator
from custom_layers.pyramidal_processor import PyramidalProcessor
from custom_layers.upsample_layer import UpsampleLayer
from custom_layers.warp_layer import WarpLayer


class Model(tf.keras.Model):
    def __init__(self, name='PWC-net', **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

        self.pyramidal_processor = PyramidalProcessor()
        self.cost_volume_layer = CostVolumeLayer()
        self.flow_estimator = FlowEstimator()
        self.warp_layer = WarpLayer()
        self.upsample_layer = UpsampleLayer()

    def call(self, inputs):
        flows = []

        first_frame_pyramid = self.pyramidal_processor(inputs[0])
        second_frame_pyramid = self.pyramidal_processor(inputs[1])

        old_flow_features = None
        old_flow = None

        for lvl in range(5, -1, -1):
            first_frame_features = first_frame_pyramid[lvl]
            second_frame_features = second_frame_pyramid[lvl]
            features, flow = self.computation_step(
                first_frame_features=first_frame_features,
                second_frame_features=second_frame_features,
                old_flow_features=old_flow_features,
                old_flow=old_flow)
            flows.append(flow)

        return flows

    def computation_step(self, first_frame_features, second_frame_features, old_flow_features, old_flow):
        if old_flow is None:
            cost_vol_output = self.cost_volume_layer(first_frame_features, second_frame_features)
            flow_estimator_input = cost_vol_output
        else:
            warp_output = self.warp_layer(second_frame_features, old_flow)
            cost_vol_output = self.cost_volume_layer(first_frame_features, warp_output, 4)
            flow_estimator_input = concatenate([old_flow_features, old_flow, cost_vol_output, second_frame_features])

        new_flow_features, new_flow = self.flow_estimator(flow_estimator_input)

        features = self.upsample_layer(new_flow_features)
        flow = self.upsample_layer(new_flow)

        return features, flow
