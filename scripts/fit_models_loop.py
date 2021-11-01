"""Loop over multiple model types/sets of data."""

import os
import subprocess
import argparse
import shutil
import yaml

from daart_utils.paths import data_path, config_path, results_path

# assumes `fit_models_loop.py` and `fit_models.py` are in the same directory
grid_search_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fit_models.py')


def run_main(args):

    if args.dataset == 'fly':
        # import experiment ids from file
        from daart_utils.session_ids.fly import SESS_IDS_TRAIN_5
        sess_ids_list = SESS_IDS_TRAIN_5
        # set config files
        config_files = {
            'data': os.path.join(config_path, 'data_fly.yaml'),
            'model': os.path.join(config_path, 'model.yaml'),
            'train': os.path.join(config_path, 'train.yaml')
        }
    elif args.dataset == 'ibl':
        # import experiment ids from file
        from daart_utils.session_ids.ibl import SESS_IDS_TRAIN_5
        sess_ids_list = SESS_IDS_TRAIN_5
        config_files = {
            'data': os.path.join(config_path, 'data_ibl.yaml'),
            'model': os.path.join(config_path, 'model.yaml'),
            'train': os.path.join(config_path, 'train.yaml')
        }
    else:
        raise NotImplementedError('"%s" is an invalid dataset' % args.dataset)

    # create temporary config files (will be updated each iteration, then deleted)
    configs_to_update = ['data', 'model', 'train']
    for config in configs_to_update:
        dirname = os.path.dirname(config_files[config])
        filename = os.path.basename(config_files[config]).split('.')[0]
        tmp_file = os.path.join(dirname, filename + '_tmp.yaml')
        shutil.copy(config_files[config], tmp_file)
        config_files[config] = tmp_file

    # get list of models
    model_types = []
    if args.fit_mlp:
        model_types.append('temporal-mlp')
    if args.fit_dtcn:
        model_types.append('dtcn')
    if args.fit_lstm:
        model_types.append('lstm')
    if args.fit_gru:
        model_types.append('gru')

    for model_type in model_types:

        for sess_ids in sess_ids_list:

            # modify configs
            update_config(config_files['model'], 'model_type', model_type)
            update_config(config_files['data'], 'expt_ids', sess_ids)
            update_config(config_files['data'], 'data_dir', os.path.join(data_path, args.dataset))
            update_config(config_files['data'], 'results_dir', os.path.join(results_path, args.dataset))

            call_str = [
                'python',
                grid_search_file,
                '--data_config', config_files['data'],
                '--model_config', config_files['model'],
                '--train_config', config_files['train']
            ]
            subprocess.call(' '.join(call_str), shell=True)

    for config in configs_to_update:
        os.remove(config_files[config])


def update_config(file, key, value):

    # load yaml file as dict
    config = yaml.safe_load(open(file))

    # update value
    config[key] = value

    # resave file
    with open(file, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
                          
    """To fit, for example, dtcn models on the IBL data:

    (daart) $: python fit_models_loop.py --dataset ibl --fit_dtcn

    The details of the hyperparameter search will be defined in the user config files.

    """

    parser = argparse.ArgumentParser()

    # define dataset to fit: 'fly', 'ibl'
    parser.add_argument('--dataset')

    # define models to run
    parser.add_argument('--fit_mlp', action='store_true', default=False)
    parser.add_argument('--fit_lstm', action='store_true', default=False)
    parser.add_argument('--fit_gru', action='store_true', default=False)
    parser.add_argument('--fit_dtcn', action='store_true', default=False)

    namespace, _ = parser.parse_known_args()
    run_main(namespace)
