from config import CONFIG
from dataset_loaders.chairs_dataset_loader import ChairsDatasetLoader
from logger import Logger
from model import Model
from model_trainer import ModelTrainer
from utils.config_utils import initialize_results_dir

config = initialize_results_dir(CONFIG)
loader = ChairsDatasetLoader(config)
model = Model(config['model'])
logger = Logger(config)
trainer = ModelTrainer(config, loader, model, logger)
trainer.train(n_steps=10000)
