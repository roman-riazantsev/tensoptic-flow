import tensorflow as tf
from tensorflow.keras.layers import concatenate, Conv2DTranspose
from custom_layers.cost_volume_layer import CostVolumeLayer
from custom_layers.feature_extractor import FeatureExtractor
from custom_layers.flow_estimator import FlowEstimator
from custom_layers.flow_refiner import FlowRefiner
from custom_layers.upsamling_layer import UpsamplingLayer
from custom_layers.warp_layer import WarpLayer


# noinspection PyCallingNonCallable
class Model(tf.keras.Model):
    def __init__(self, config, name='PWC-net'):
        super(Model, self).__init__(name=name)

        pyramid_levels = config['pyramid_levels']
        upsampling_type = config['upsampling_type']
        self.comp_depth = config['comp_depth']

        self.feature_extractor = FeatureExtractor(pyramid_levels)
        self.warp_layer = WarpLayer()
        self.cost_volume_layer = CostVolumeLayer()
        self.features_upsamplers = [UpsamplingLayer(upsampling_type) for _ in range(self.comp_depth)]
        self.flow_upsamplers = [UpsamplingLayer(upsampling_type) for _ in range(self.comp_depth)]
        self.flow_estimators = [FlowEstimator() for _ in range(self.comp_depth)]
        self.flow_refiner = FlowRefiner()

    def call(self, inputs):
        flows = []
        features_list = []

        first_frame_pyramid = self.feature_extractor(inputs[0])
        second_frame_pyramid = self.feature_extractor(inputs[1])

        old_flow_features, old_flow = None, None

        for flow_lvl in range(self.comp_depth):
            frames_features = first_frame_pyramid.pop(), second_frame_pyramid.pop()

            flow_features, flow = self.computation_step(
                frames_features=frames_features,
                old_flow_features=old_flow_features,
                old_flow=old_flow,
                flow_lvl=flow_lvl
            )
            features_list.append(flow_features)
            flows.append(flow)
            old_flow_features, old_flow = flow_features, flow

        last_features, last_flow = features_list[-1], flows[-1]
        refined_flow = self.flow_refiner(last_features, last_flow)
        flows.append(refined_flow)
        return flows

    def computation_step(self, frames_features, old_flow_features, old_flow,
                         flow_lvl):
        first_frame_features, second_frame_features = frames_features
        flow_estimator = self.flow_estimators[flow_lvl]
        feature_upsamler = self.features_upsamplers[flow_lvl]
        flow_upsamler = self.flow_upsamplers[flow_lvl]

        if old_flow is None:
            cost_vol_output = self.cost_volume_layer(first_frame_features, second_frame_features)
            flow_estimator_input = cost_vol_output
        else:
            warp_output = self.warp_layer(second_frame_features, old_flow)
            cost_vol_output = self.cost_volume_layer(first_frame_features, warp_output, 4)
            flow_estimator_input = concatenate([old_flow_features, old_flow, cost_vol_output, second_frame_features])

        new_flow_features, new_flow = flow_estimator(flow_estimator_input)
        features = feature_upsamler(new_flow_features)
        flow = flow_upsamler(new_flow)
        return features, flow
