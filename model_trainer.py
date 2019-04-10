import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np

from utils import pad_batch


class ModelTrainer(object):
    def __init__(self, config, loader, model):
        self.config = config
        self.loader = loader
        self.model = model

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            frame_1, frame_2, flow = self.loader.next_batch()

            dimensions = [(8, 6), (16, 12), (32, 24), (65, 48), (128, 96), (256, 192)]

            resized_flows = []

            for i in dimensions:
                resized_flow = []
                for img in flow:
                    img = cv2.resize(img, i)
                    img[..., 0] /= (i[0] / self.config['img_width'])
                    img[..., 1] /= (i[1] / self.config['img_height'])
                    resized_flow.append(img)
                resized_flow = np.array(resized_flow)
                resized_flows.append(resized_flow)

            total_loss = self.train_step(frame_1, frame_2, resized_flows)
            # Log every 10 batches.
            if epoch % 10 == 0:
                print('Training loss at step {}: {}'.format(epoch, np.array(total_loss).mean()))
                print('Seen so far: {} samples'.format((epoch + 1) * 1))

    def train_step(self, frame_1, frame_2, resized_flows):
        with tf.GradientTape() as tape:
            pred_flows = self.model([frame_1, frame_2])  # Logits for this minibatchv
            losses = []

            for true_flow, pred_flow in zip(resized_flows, pred_flows):
                true_flow_paded = pad_batch(true_flow, data_type='numpy')
                pred_flow_paded = pad_batch(pred_flow, data_type='tensor')

                loss_value = keras.losses.mse(true_flow_paded, pred_flow_paded)
                losses.append(loss_value)

            total_loss = sum(losses)

            grads = tape.gradient(total_loss, self.model.trainable_variables)

            optimizer = keras.optimizers.Adam()
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            return total_loss
