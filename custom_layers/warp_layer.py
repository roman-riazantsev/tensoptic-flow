"""
Based on:
    - https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/core_warp.py
      Written by Phil Ferriere
"""

from __future__ import absolute_import, division, print_function
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class WarpLayer(layers.Layer):
    def __init__(self):
        super(WarpLayer, self).__init__()

    def call(self, image, flow):
        # print(image, flow)
        # print(image.shape)
        batch_size, height, width, channels = tf.unstack(image.shape)

        w = math_ops.range(width)
        h = math_ops.range(height)

        grid_x, grid_y = array_ops.meshgrid(w, h)
        stacked_grid = math_ops.cast(array_ops.stack([grid_y, grid_x], axis=2), tf.float32)
        batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
        grid_flow = batched_grid - flow

        flattened = array_ops.reshape(grid_flow,
                                      [batch_size, height * width, 2])
        interpolated = self._interpolate_bilinear(image, flattened)
        interpolated = array_ops.reshape(interpolated,
                                         [batch_size, height, width, channels])

        return interpolated

    @staticmethod
    def _interpolate_bilinear(grid,
                              query_points,
                              name='interpolate_bilinear',
                              indexing='ij'):
        """Similar to Matlab's interp2 function.

        Finds values for query points on a grid using bilinear interpolation.

        Args:
          grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
          query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
          name: a name for the operation (optional).
          indexing: whether the query points are specified as row and column (ij),
            or Cartesian coordinates (xy).

        Returns:
          values: a 3-D `Tensor` with shape `[batch, N, channels]`

        Raises:
          ValueError: if the indexing mode is invalid, or if the shape of the inputs
            invalid.
        """
        if indexing != 'ij' and indexing != 'xy':
            raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

        with ops.name_scope(name):
            grid = ops.convert_to_tensor(grid)
            query_points = ops.convert_to_tensor(query_points)
            shape = array_ops.unstack(array_ops.shape(grid))
            if len(shape) != 4:
                msg = 'Grid must be 4 dimensional. Received: '
                raise ValueError(msg + str(shape))

            batch_size, height, width, channels = shape
            query_type = query_points.dtype
            query_shape = array_ops.unstack(array_ops.shape(query_points))
            grid_type = grid.dtype

            if len(query_shape) != 3:
                msg = ('Query points must be 3 dimensional. Received: ')
                raise ValueError(msg + str(query_shape))

            _, num_queries, _ = query_shape

            alphas = []
            floors = []
            ceils = []

            index_order = [0, 1] if indexing == 'ij' else [1, 0]
            unstacked_query_points = array_ops.unstack(query_points, axis=2)

            for dim in index_order:
                with ops.name_scope('dim-' + str(dim)):
                    queries = unstacked_query_points[dim]

                    size_in_indexing_dimension = shape[dim + 1]

                    # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                    # is still a valid index into the grid.
                    max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
                    min_floor = constant_op.constant(0.0, dtype=query_type)
                    floor = math_ops.minimum(
                        math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                    int_floor = math_ops.cast(floor, dtypes.int32)
                    floors.append(int_floor)
                    ceil = int_floor + 1
                    ceils.append(ceil)

                    # alpha has the same type as the grid, as we will directly use alpha
                    # when taking linear combinations of pixel values from the image.
                    alpha = math_ops.cast(queries - floor, grid_type)
                    min_alpha = constant_op.constant(0.0, dtype=grid_type)
                    max_alpha = constant_op.constant(1.0, dtype=grid_type)
                    alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

                    # Expand alpha to [b, n, 1] so we can use broadcasting
                    # (since the alpha values don't depend on the channel).
                    alpha = array_ops.expand_dims(alpha, 2)
                    alphas.append(alpha)

            flattened_grid = array_ops.reshape(grid,
                                               [batch_size * height * width, channels])
            batch_offsets = array_ops.reshape(
                math_ops.range(batch_size) * height * width, [batch_size, 1])

            # This wraps array_ops.gather. We reshape the image data such that the
            # batch, y, and x coordinates are pulled into the first dimension.
            # Then we gather. Finally, we reshape the output back. It's possible this
            # code would be made simpler by using array_ops.gather_nd.
            def gather(y_coords, x_coords, name):
                with ops.name_scope('gather-' + name):
                    linear_coordinates = batch_offsets + y_coords * width + x_coords
                    gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                    return array_ops.reshape(gathered_values,
                                             [batch_size, num_queries, channels])

            # grab the pixel values in the 4 corners around each query point
            top_left = gather(floors[0], floors[1], 'top_left')
            top_right = gather(floors[0], ceils[1], 'top_right')
            bottom_left = gather(ceils[0], floors[1], 'bottom_left')
            bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

            # now, do the actual interpolation
            with ops.name_scope('interpolate'):
                interp_top = alphas[1] * (top_right - top_left) + top_left
                interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
                interp = alphas[0] * (interp_bottom - interp_top) + interp_top

            return interp
