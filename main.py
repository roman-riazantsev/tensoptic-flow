from dataset_loaders.chairs_dataset_loader import ChairsDatasetLoader
from model import Model

from model import Model
from model_trainer import ModelTrainer

config = {
    'dataset_dir_path': "../../Datasets/FlyingChairs_release/data",
    'img_height': 192,
    'img_width': 256,
    'batch_size': 8,
    'learning_rate': 0.0001,
    'save_rate': 10,
    'save_path': 'saves/'
}

loader = ChairsDatasetLoader(config)
model = Model()
trainer = ModelTrainer(config, loader, model)
trainer.train(n_steps=2000)
