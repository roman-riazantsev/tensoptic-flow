CONFIG = {
    'dataset_dir_path': "../../Datasets/FlyingChairs_release/data",
    'img_height': 384,
    'img_width': 512,
    'batch_size': 1,
    'learning_rate': 0.0001,
    'save_rate': 10,
    'train_set_size': 3,
    'loss_normalizations': [0.32, 0.08, 0.02, 0.01, 0.005],
    'model': {
        'model_type': 'PWC',
        'pyramid_levels': 5,
        'comp_depth': 4,
        'upsampling_type': 'Conv2DTranspose'  # 'Conv2DTranspose' or 'UpSampling2D'
    }
}
