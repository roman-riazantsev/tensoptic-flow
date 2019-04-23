import tensorflow as tf
import numpy as np


def pad_batch(batch, data_type, max_in_dims=[48, 65], constant_values=0):
    if data_type == 'numpy':
        default_shape = batch.shape
    else:
        default_shape = tf.shape(batch).numpy()

    default_dims = default_shape[1:3]

    paddings = [[0, 0]]

    for (default_dim, padded_dim) in zip(default_dims, max_in_dims):
        dim_diff = padded_dim - default_dim
        first_side_pad = int(dim_diff / 2)
        second_side_pad = dim_diff - first_side_pad
        paddings.append([first_side_pad, second_side_pad])

    paddings.append([0, 0])

    if data_type == 'numpy':
        result = np.pad(batch, paddings, 'constant', constant_values=constant_values)
    else:
        result = tf.pad(batch, paddings, 'CONSTANT', constant_values=constant_values)

    return result
