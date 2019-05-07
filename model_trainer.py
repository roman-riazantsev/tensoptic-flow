import os

import tensorflow as tf
import tensorflow.keras as keras
import pickle


class ModelTrainer(object):
    def __init__(self, config, loader, model, logger):
        self.logger = logger
        self.loader = loader
        self.model = model
        self.step = 0

        self.loss_normalizations = config['loss_normalizations']
        self.learning_rate = config['learning_rate']
        self.saved_step_path = config['saved_step_path']
        self.saves_dir_path = config['saves_dir_path']
        self.save_rate = config['save_rate']

    def train(self, n_steps):
        self.load_model()
        self.step += 1

        for step in range(n_steps):
            frame_1, frame_2, flow = self.loader.next_batch()

            scaled_flows_gt, pred_flows = self.train_step(frame_1, frame_2, flow)

            template = 'Step {}, Loss: {}'
            print(template.format(self.step, self.logger.get_loss_value()))

            if self.step % self.save_rate == 0:
                self.logger.log_loss(self.step)
                self.logger.log_flows(scaled_flows_gt, pred_flows, self.step)
                self.save_model()

            self.logger.reset_loss_states()
            self.step += 1

    def train_step(self, frame_1, frame_2, flow_gt):
        with tf.GradientTape() as tape:
            pred_flows = self.model([frame_1, frame_2])
            scaled_flows_gt = []
            total_loss = 0.
            loss_normalizations = [0.32, 0.08, 0.02, 0.01, 0.005]

            for pred_flow, loss_norm in zip(pred_flows, loss_normalizations):
                _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(pred_flow))

                scaled_flow_gt = tf.image.resize_with_pad(flow_gt, lvl_height, lvl_width)
                scaled_flow_gt /= tf.cast(flow_gt.shape[1] / lvl_height, dtype=tf.float32)
                scaled_flows_gt.append(scaled_flow_gt)

                normed_loss = tf.norm(scaled_flow_gt - pred_flow, ord=2, axis=3)
                lvl_loss = tf.reduce_mean(tf.reduce_sum(normed_loss, axis=(1, 2)))
                total_loss += lvl_loss * loss_norm

            gamma = 0.0004
            loss_regularization = gamma * tf.reduce_sum([tf.nn.l2_loss(var) for var in self.model.trainable_weights])
            total_loss += loss_regularization
            last_flow_loss = tf.reduce_mean(tf.norm(scaled_flows_gt[-1] - pred_flows[-1], ord=2, axis=3))

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer = keras.optimizers.Adam(lr=self.learning_rate)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.logger.loss_metric(total_loss)
            self.logger.final_loss_metric(last_flow_loss)

            return scaled_flows_gt, pred_flows

    def save_model(self):
        self.model.save_weights(self.saves_dir_path, save_format='tf')
        with open(self.saved_step_path, 'wb') as f:
            pickle.dump(self.step, f)

    def load_model(self):
        if len(os.listdir(self.saves_dir_path)) <= 1:
            print('No saves in folder')
        else:
            self.model.load_weights(self.saves_dir_path)
            with open(self.saved_step_path, 'rb') as f:
                self.step = pickle.load(f)
