import glob
import os
import numpy as np
import cv2

from dataset_loaders.base_loader import BaseLoader


class ChairsDatasetLoader(BaseLoader):
    def get_dataset_info(self):
        train_set_size = 0
        test_set_size = 0

        def get_examples_paths(feature_folder):
            nonlocal train_set_size
            nonlocal test_set_size

            feature_paths = sorted(glob.glob(os.path.join(self.dataset_dir_path, feature_folder, "*")))

            if train_set_size == 0:
                dataset_size = len(feature_paths)
                train_set_size = self.train_set_size
                test_set_size = dataset_size - train_set_size


            test_paths = feature_paths[train_set_size:]
            train_paths = feature_paths[:train_set_size - 1]

            return {"train": train_paths, "test": test_paths}

        features_info = [{
            'processing_function': self.process_img,
            'depth': 3,
            'folder': 'img1'
        }, {
            'processing_function': self.process_img,
            'depth': 3,
            'folder': 'img2'
        }, {
            'processing_function': self.decode_flow,
            'depth': 2,
            'folder': 'flow'
        }]

        for feature_info in features_info:
            feature_folder = feature_info['folder']
            feature_info['paths'] = get_examples_paths(feature_folder)

        return features_info, train_set_size, test_set_size

    def decode_flow(self, flow_path, depth=2):
        f = open(flow_path, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception("Flow file does not contain PIEH header.")

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * depth).reshape((height, width, depth))

        flow = cv2.resize(flow, (self.img_width, self.img_height))
        flow[..., 0] /= (self.img_width / width)
        flow[..., 1] /= (self.img_height / height)
        return flow
