from dataset_loaders.chairs_dataset_loader import ChairsDatasetLoader
from model import Model

# 1) configure
from model_2 import Model2
from model_3 import Model3
from model_4 import Model4
from model_trainer import ModelTrainer
from model_trainer_2 import ModelTrainer2
from model_trainer_3 import ModelTrainer3
from trainers.trainer_lvl_1 import TrainerLVL1

config = {
    'dataset_dir_path': "../../Datasets/FlyingChairs_release/data",
    'img_height': 192,
    'img_width': 256,
    'batch_size': 8,
    'learning_rate': 0.0001,
    'save_rate': 10,
    'save_path': 'saves/'
}
# 2) create dataset loader
loader = ChairsDatasetLoader(config)
# 3) build model
model = Model4()
# 4) run
trainer = ModelTrainer3(config, loader, model)
trainer.train(n_steps=1000)
