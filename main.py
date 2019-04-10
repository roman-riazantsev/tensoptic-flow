from dataset_loaders.chairs_dataset_loader import ChairsDatasetLoader
from model import Model

# 1) configure
config = {
    'dataset_dir_path': "../../Datasets/FlyingChairs_release/data",
    'img_height': 192,
    'img_width': 256,
    'batch_size': 2,
}
# 2) create dataset loader
loader = ChairsDatasetLoader(config)
# 3) build model
model = Model()
# 4) run
