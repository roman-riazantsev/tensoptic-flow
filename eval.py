from config import CONFIG
from dataset_loaders.chairs_dataset_loader import ChairsDatasetLoader
from logger import Logger
from model import Model
from model_evalutaor import ModelEvaluator
from utils.config_utils import initialize_results_dir

CONFIG = {
    'dataset_dir_path': "../datasets/FlyingChairs/FlyingChairs_release/data",
    'img_height': 384,
    'img_width': 512,
    'batch_size': 1,
    'train_set_size': 7000,
    'n_steps': 1000,
    'model': {
        'model_type': 'PWC',
        'pyramid_levels': 5,
        'comp_depth': 4,
        'upsampling_type': 'Conv2DTranspose'  # 'Conv2DTranspose' or 'UpSampling2D'
    }
}

config = initialize_results_dir(CONFIG)
loader = ChairsDatasetLoader(config)
model = Model(config['model'])
trainer = ModelEvaluator(config, loader, model)
trainer.evaluate(n_steps=config['n_steps'])
