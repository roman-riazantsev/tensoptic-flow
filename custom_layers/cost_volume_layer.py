"""
Based on:
    - https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/core_costvol.py
      Written by Phil Ferriere
"""

import tensorflow as tf
from tensorflow.keras import layers


class CostVolumeLayer(layers.Layer):
    def __init__(self):
        super(CostVolumeLayer, self).__init__()

    @staticmethod
    def call(name, c1, warp, search_range):
        """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
            Args:
                c1: Level of the feature pyramid of Image1
                warp: Warped level of the feature pyramid of image22
                search_range: Search range (maximum displacement)
            """
        padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
        _, h, w, _ = tf.unstack(tf.shape(c1))
        max_offset = search_range * 2 + 1

        cost_vol = []
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
                cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
                cost_vol.append(cost)
        cost_vol = tf.concat(cost_vol, axis=3)
        cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

        return cost_vol
