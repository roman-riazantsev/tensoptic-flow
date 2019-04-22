from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import flow_vis
from dataset_loaders.chairs_dataset_loader import ChairsDatasetLoader
from model import Model
from model_evaluator import ModelEvaluator

config = {
    'dataset_dir_path': "../../Datasets/FlyingChairs_release/data",
    'img_height': 192,
    'img_width': 256,
    'batch_size': 1,
    'learning_rate': 0.0001,
    'save_rate': 10,
    'save_path': 'saves/'
}

loader = ChairsDatasetLoader(config)
model = Model()

# 1 load weights
evaluator = ModelEvaluator(config, loader, model)
# 2 run on random image
evaluator.evaluate()
# 3 visualize ground truth and predicted


