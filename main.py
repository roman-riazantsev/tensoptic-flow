from dataset_loaders.chairs_dataset_loader import ChairsDatasetLoader
from model import Model
from tensorflow import keras

# 1) configure
from model_trainer import ModelTrainer

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
trainer = ModelTrainer(config, loader, model)
trainer.train(n_epochs=1000)
