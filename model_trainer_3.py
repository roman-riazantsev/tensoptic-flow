import os

import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np

from utils import pad_batch


class ModelTrainer3(object):
    def __init__(self, config, loader, model):
        self.config = config
        self.loader = loader
        self.model = model

        self.loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.log_dir = 'logs/gradient_tape/PWC/train'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.save_rate = config['save_rate']

    def train(self, n_steps):
        self.load_model()

        for step in range(n_steps):
            frame_1, frame_2, flow = self.loader.next_batch()

            dimensions = [(8, 6), (16, 12), (32, 24), (64, 48)]

            resized_flows = []

            for dim in dimensions:
                resized_flow = []
                for img in flow:
                    img = cv2.resize(img, dim)
                    img[..., 0] /= (dim[0] / self.config['img_width'])
                    img[..., 1] /= (dim[1] / self.config['img_height'])
                    resized_flow.append(img)
                resized_flow = np.array(resized_flow)
                resized_flows.append(resized_flow)

            resized_flows.append(resized_flows[-1])

            self.train_step(frame_1, frame_2, resized_flows)

            with self.summary_writer.as_default():
                tf.summary.scalar('loss', self.loss_metric.result(), step=step)

            template = 'Step {}, Loss: {}'
            print(template.format(step + 1, self.loss_metric.result()))

            if step % self.save_rate == 0:
                self.save_model()

            self.loss_metric.reset_states()

    def train_step(self, frame_1, frame_2, resized_flows):
        with tf.GradientTape() as tape:
            pred_flows = self.model([frame_1, frame_2])
            total_loss = 0.
            loss_normalizations = [0.32, 0.08, 0.02, 0.01, 0.005]

            for true_flow, pred_flow, loss_norm in zip(resized_flows, pred_flows, loss_normalizations):
                normed_loss = tf.norm(true_flow - pred_flow, ord=2, axis=3)
                lvl_loss = tf.reduce_mean(tf.reduce_sum(normed_loss, axis=(1, 2)))
                total_loss += lvl_loss * loss_norm

            gamma = 0.0004
            loss_regularization = gamma * tf.reduce_sum([tf.nn.l2_loss(var) for var in self.model.trainable_weights])
            total_loss += loss_regularization

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer = keras.optimizers.Adam(lr=self.config['learning_rate'])
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.loss_metric(total_loss)

    def save_model(self):
        self.model.save_weights(self.config['save_path'], save_format='tf')

    def load_model(self):
        if os.path.exists(self.config['save_path']):
            self.model.load_weights(self.config['save_path'])
        else:
            os.makedirs(self.config['save_path'])
