from dataset_loaders.chairs_dataset_loader import ChairsDatasetLoader
from model import Model
from tensorflow import keras

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
# model_check
first_frame_input = keras.Input(shape=(config['img_height'], config['img_width'], 3), batch_size=config['batch_size'],
                                name='frame_1')
second_frame_input = keras.Input(shape=(config['img_height'], config['img_width'], 3), batch_size=config['batch_size'],
                                 name='frame_2')
fram1, fram2, flow = loader.next_batch()
flows = model([fram1, fram2])
for flow in flows:
    print(flow.shape)
# 4) run
