import os

import tensorflow as tf
import tensorflow.keras as keras
import pickle


class ModelTrainer(object):
    def __init__(self, config, loader, model):
        self.config = config
        self.loader = loader
        self.model = model
        self.step = 0

        self.path_step = os.path.join(self.config['save_path'], 'save_path')

        self.loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.log_dir = 'logs/gradient_tape/PWC/train'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.save_rate = config['save_rate']

    def train(self, n_steps):
        self.load_model()
        self.step += 1

        for step in range(n_steps):
            frame_1, frame_2, flow = self.loader.next_batch()

            self.train_step(frame_1, frame_2, flow)

            template = 'Step {}, Loss: {}'
            print(template.format(self.step, self.loss_metric.result()))

            if self.step % self.save_rate == 0:
                with self.summary_writer.as_default():
                    tf.summary.scalar('loss', self.loss_metric.result(), step=self.step)
                self.save_model()

            self.loss_metric.reset_states()
            self.step += 1

    def train_step(self, frame_1, frame_2, flow_gt):
        with tf.GradientTape() as tape:
            pred_flows = self.model([frame_1, frame_2])
            total_loss = 0.
            loss_normalizations = [0.32, 0.08, 0.02, 0.01, 0.005]

            for pred_flow, loss_norm in zip(pred_flows, loss_normalizations):
                _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(pred_flow))

                scaled_flow_gt = tf.image.resize_with_pad(flow_gt, lvl_height, lvl_width)
                scaled_flow_gt /= tf.cast(flow_gt.shape[1] / lvl_height, dtype=tf.float32)

                normed_loss = tf.norm(scaled_flow_gt - pred_flow, ord=2, axis=3)
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
        with open(self.path_step, 'wb') as f:
            pickle.dump(self.step, f)

    def load_model(self):
        if os.path.exists(self.config['save_path']):
            if len(os.listdir(self.config['save_path'])) != 0:
                self.model.load_weights(self.config['save_path'])
                with open(self.path_step, 'rb') as f:
                    self.step = pickle.load(f)
        else:
            os.makedirs(self.config['save_path'])
