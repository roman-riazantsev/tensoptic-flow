import os

import tensorflow as tf
import tensorflow.keras as keras
import pickle
import random


class ModelEvaluator(object):
    def __init__(self, config, loader, model):
        self.loader = loader
        self.model = model

        self.saved_step_path = config['saved_step_path']
        self.saves_dir_path = config['saves_dir_path']

        self.subset_size = self.loader.test_set_size

    def evaluate(self, n_steps):
        self.load_model()

        for step in range(n_steps):
            idx = random.randint(0, self.subset_size)

            frame_1, frame_2, flow_gt = self.loader.next_batch(subset="test")
            frame_1, frame_2, flow_gt = self.loader.get_observation(idx, subset="test")

            pred_flow = self.model([frame_1, frame_2])[-1]
            _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(pred_flow))
            scaled_flow_gt = tf.image.resize_with_pad(flow_gt, lvl_height, lvl_width)
            scaled_flow_gt /= tf.cast(flow_gt.shape[1] / lvl_height, dtype=tf.float32)
            loss = tf.reduce_mean(tf.norm(scaled_flow_gt - pred_flow, ord=2, axis=3))

            template = 'Step {}, Idx: {}, Loss: {}'
            print(template.format(step, idx, loss))

            step += 1

            # Calculate losses for all images.

        # Visualize top worst
        # Visualize top bad results
        # Visualize average results

    def load_model(self):
        if len(os.listdir(self.saves_dir_path)) <= 1:
            print('No saves in folder')
        else:
            self.model.load_weights(self.saves_dir_path)

    # def save_pair(self):
