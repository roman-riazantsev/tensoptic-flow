import cv2
import numpy as np
import random


class BaseLoader(object):
    def __init__(self, config):
        self.__dict__.update(config)

        self.features_info, self.train_set_size, self.test_set_size = self.get_dataset_info()

    def get_dataset_info(self):
        return {}, 0, 0

    def next_batch(self, subset="train"):
        batch = []

        for feature_info in self.features_info:
            feature_batch = np.empty([self.batch_size, self.img_height, self.img_width, feature_info['depth']],
                                     dtype=np.float32)
            batch.append(feature_batch)

        subset_size = self.train_set_size if subset == "train" else self.test_set_size

        for i in range(self.batch_size):
            idx = random.randint(0, subset_size - 2)

            for feature_number, feature_info in enumerate(self.features_info):
                process_data = feature_info['processing_function']
                path = feature_info['paths'][subset][idx]
                depth = feature_info['depth']
                batch[feature_number][i] = process_data(path, depth)

        return batch

    def get_observation(self, idx, subset="test"):
        observation = []

        for feature_info in self.features_info:
            feature = np.empty([self.batch_size, self.img_height, self.img_width, feature_info['depth']],
                               dtype=np.float32)
            observation.append(feature)

        for feature_number, feature_info in enumerate(self.features_info):
            process_data = feature_info['processing_function']
            path = feature_info['paths'][subset][idx]
            depth = feature_info['depth']
            observation[feature_number][0] = process_data(path, depth)

        return observation

    def process_img(self, path_to_image, depth):
        if depth == 1:
            is_rgb = False
        else:
            is_rgb = True
        img = cv2.imread(path_to_image, is_rgb)
        img = cv2.resize(img, (self.img_width, self.img_height)).reshape([self.img_height, self.img_width, -1])
        return img / 255
