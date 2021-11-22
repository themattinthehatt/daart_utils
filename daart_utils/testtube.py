"""Utility functions for fitting models from the command line with testtube."""

import os
import sys
import yaml

from daart.io import export_expt_info_to_csv
from daart.io import find_experiment
from daart.io import get_expt_dir
from daart.io import get_model_dir
from daart.io import get_subdirs


def get_all_params():
    # raise error if user has other command line arguments specified
    if len(sys.argv[1:]) != 6:
        raise ValueError('No command line arguments allowed other than config file names')

    def add_to_parser(parser, arg_name, value):
        if arg_name == 'expt_ids':
            # treat expt_ids differently, want to parse full lists as one
            if isinstance(value, list):
                value = ';'.join(value)
            parser.add_argument('--' + arg_name, default=value)
        elif isinstance(value, list):
            parser.opt_list('--' + arg_name, options=value, tunable=True)
        else:
            parser.add_argument('--' + arg_name, default=value)

    # create parser
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--train_config', type=str)

    namespace, extra = parser.parse_known_args()

    # add arguments from all configs
    configs = [namespace.data_config, namespace.model_config, namespace.train_config]
    for config in configs:
        config_dict = yaml.safe_load(open(config))
        for (key, value) in config_dict.items():
            add_to_parser(parser, key, value)

    return parser.parse_args()


def print_hparams(hparams):
    """Pretty print hparams to console."""
    config_files = ['data', 'model', 'train']
    for config_file in config_files:
        print('\n%s CONFIG:' % config_file.upper())
        config_dict = yaml.safe_load(open(hparams['%s_config' % config_file]))
        for key in config_dict.keys():
            print('    {}: {}'.format(key, hparams[key]))
    print('')


def create_tt_experiment(hparams):
    """Create test-tube experiment for organizing model fits.

    Parameters
    ----------
    hparams : dict
        dictionary of hyperparameters defining experiment

    Returns
    -------
    tuple
        - if experiment defined by hparams already exists, returns `(None, None)`
        - if experiment does not exist, returns `(hparams, exp)`

    """
    from test_tube import Experiment

    # get model path
    if not os.path.isdir(hparams['results_dir']):
        os.makedirs(hparams['results_dir'])
    hparams['expt_dir'] = get_expt_dir(hparams['results_dir'], hparams['expt_ids'])
    if not os.path.isdir(hparams['expt_dir']):
        os.makedirs(hparams['expt_dir'])
        export_expt_info_to_csv(hparams['expt_dir'], hparams['expt_ids'])
    hparams['model_dir'] = get_model_dir(hparams['expt_dir'], hparams)
    tt_expt_dir = os.path.join(hparams['model_dir'], hparams['tt_experiment_name'])
    if not os.path.isdir(tt_expt_dir):
        os.makedirs(tt_expt_dir)

    # check to see if experiment already exists
    if find_experiment(hparams) is not None:
        return None, None

    exp = Experiment(
        name=hparams['tt_experiment_name'],
        debug=False,
        save_dir=os.path.dirname(hparams['model_dir']))
    hparams['version'] = exp.version
    hparams['tt_version_dir'] = os.path.join(tt_expt_dir, 'version_%i' % exp.version)

    return hparams, exp


def clean_tt_dir(hparams):
    """Delete all (unnecessary) subdirectories in the model directory (created by test-tube)"""
    import shutil
    # get subdirs
    version_dir = hparams['tt_version_dir']
    subdirs = get_subdirs(version_dir)
    for subdir in subdirs:
        shutil.rmtree(os.path.join(version_dir, subdir))
    os.remove(os.path.join(version_dir, 'meta.experiment'))
