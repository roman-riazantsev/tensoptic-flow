import os


def initialize_results_dir(config_dict):
    model_type = config_dict['model']['model_type']
    pyramid_levels = config_dict['model']['pyramid_levels']
    upsampling_type = config_dict['model']['upsampling_type']
    assert pyramid_levels >= (config_dict['model']['comp_depth']), "Computed flow larger then frames."
    config_dict['build_id'] = '{}_{}_lvls_{}'.format(model_type, pyramid_levels, upsampling_type)
    print("BUILD_ID: {}".format(config_dict['build_id']))

    results_root_name = config_dict['build_id'] + '_results'
    results_root = os.path.join('results', results_root_name + '/')
    results_dirs_paths = {
        'results_root': results_root,
        'evaluation_dir_path': os.path.join(results_root, 'evaluation' + '/'),
        'logs_dir_path': os.path.join(results_root, 'logs' + '/'),
        'saves_dir_path': os.path.join(results_root, 'saves' + '/')
    }
    results_paths = list(results_dirs_paths.values())

    try:
        for dir_ in results_paths:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
    results_dirs_paths['saved_step_path'] = results_dirs_paths['saves_dir_path'] + 'step'
    config_dict.update(results_dirs_paths)
    return config_dict
