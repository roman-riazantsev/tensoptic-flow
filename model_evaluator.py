import os

import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import flow_vis
from utils import pad_batch


class ModelEvaluator(object):
    def __init__(self, config, loader, model):
        self.config = config
        self.loader = loader
        self.model = model

    def evaluate(self):
        self.load_model()

        frame_1, frame_2, flow_tr = self.loader.next_batch()
        flow_tr = flow_tr.reshape(192, 256, 2)

        pred_flows = self.model([frame_1, frame_2])

        flow = pred_flows[-1].numpy()

        flow = flow.reshape(192, 256, 2)
        # Apply the coloring (for OpenCV, set convert_to_bgr=True)
        flow_color_true = flow_vis.flow_to_color(flow_tr, convert_to_bgr=False)
        flow_color_pred = flow_vis.flow_to_color(flow, convert_to_bgr=False)

        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(flow_color_true)
        f.add_subplot(1, 2, 2)
        plt.imshow(flow_color_pred)
        plt.show(block=True)

    def load_model(self):
        if os.path.exists(self.config['save_path']):
            self.model.load_weights(self.config['save_path'])
        else:
            os.makedirs(self.config['save_path'])
