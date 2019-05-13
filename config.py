CONFIG = {
    'dataset_dir_path': "../datasets/FlyingChairs/FlyingChairs_release/data",
    'img_height': 384,
    'img_width': 512,
    'batch_size': 8,
    'learning_rate': 0.0001,
    'save_rate': 200,
    'img_log_rate': 1500,
    'train_set_size': 7000,
    'n_steps': 2000000,
    'loss_normalizations': [0.32, 0.08, 0.02, 0.01, 0.005],
    'model': {
        'model_type': 'PWC',
        'pyramid_levels': 5,
        'comp_depth': 4,
        'upsampling_type': 'Conv2DTranspose'  # 'Conv2DTranspose' or 'UpSampling2D'
    }
}
