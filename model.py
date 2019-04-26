import tensorflow as tf
from tensorflow.keras.layers import concatenate, Conv2DTranspose

from custom_layers.cost_volume_layer import CostVolumeLayer
from custom_layers.flow_estimator import FlowEstimator
from custom_layers.feature_extractor import FeatureExtractor
from custom_layers.flow_refiner import FlowRefiner
from custom_layers.warp_layer import WarpLayer


# noinspection PyCallingNonCallable
class Model(tf.keras.Model):
    def __init__(self, name='PWC-net'):
        super(Model, self).__init__(name=name)

        self.feature_extractor = FeatureExtractor()
        self.cost_volume_layer = CostVolumeLayer()
        self.flow_refiner = FlowRefiner()
        self.flow_estimators = [FlowEstimator() for _ in range(4)]
        self.warp_layer = WarpLayer()

    def call(self, inputs):
        flows = []
        features_list = []

        first_frame_pyramid = self.feature_extractor(inputs[0])
        second_frame_pyramid = self.feature_extractor(inputs[1])

        old_flow_features = None
        old_flow = None

        for lvl in range(3, -1, -1):
            first_frame_features = first_frame_pyramid[lvl]
            second_frame_features = second_frame_pyramid[lvl]
            flow_estimator = self.flow_estimators[lvl]

            features, flow = self.computation_step(
                first_frame_features=first_frame_features,
                second_frame_features=second_frame_features,
                old_flow_features=old_flow_features,
                old_flow=old_flow,
                flow_estimator=flow_estimator
            )
            flows.append(flow)
            features_list.append(features)

        last_features, last_flow = features_list[-1], flows[-1]

        refined_flow = self.flow_refiner(last_features, last_flow)
        flows.append(refined_flow)
        return flows

    def computation_step(self, first_frame_features, second_frame_features, old_flow_features, old_flow,
                         flow_estimator):
        if old_flow is None:
            cost_vol_output = self.cost_volume_layer(first_frame_features, second_frame_features)
            flow_estimator_input = cost_vol_output
        else:
            warp_output = self.warp_layer(second_frame_features, old_flow)
            cost_vol_output = self.cost_volume_layer(first_frame_features, warp_output, 4)
            flow_estimator_input = concatenate([old_flow_features, old_flow, cost_vol_output, second_frame_features])

        new_flow_features, new_flow = flow_estimator(flow_estimator_input)

        features = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same')(new_flow_features)
        flow = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same')(new_flow)

        return features, flow
