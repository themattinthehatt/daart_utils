"""Fit daart models (potentially searching hyperparameters on a set of data."""

import copy
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import sys
from test_tube import HyperOptArgumentParser
import time
import torch
import yaml

from daart.data import DataGenerator
from daart.io import export_hparams
from daart.transforms import ZScore
from daart.utils import compute_batch_pad

from daart_utils.testtube import get_all_params, print_hparams, create_tt_experiment, clean_tt_dir


def run_main(hparams, *args):
    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    # print hparams to console
    print_hparams(hparams)

    # start at random times (so test tube creates separate folders)
    t = time.time()
    np.random.seed(int(100000000000 * t) % (2 ** 32 - 1))
    time.sleep(np.random.uniform(2))

    # create test-tube experiment
    hparams['expt_ids'] = hparams['expt_ids'].split(';')
    hparams, exp = create_tt_experiment(hparams)
    if hparams is None:
        print('Experiment exists! Aborting fit')
        return

    # where model results will be saved
    model_save_path = hparams['tt_version_dir']

    # -------------------------------------
    # build data generator
    # -------------------------------------
    signals = []
    transforms = []
    paths = []

    for expt_id in hparams['expt_ids']:

        # DLC markers or features (i.e. from simba)
        input_type = hparams.get('input_type', 'markers')
        markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.h5')
        if not os.path.exists(markers_file):
            markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.csv')
        if not os.path.exists(markers_file):
            markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.npy')
        if not os.path.exists(markers_file):
            raise FileNotFoundError('could not find marker file for %s' % expt_id)

        # heuristic labels
        labels_file = os.path.join(
            hparams['data_dir'], 'labels-heuristic', expt_id + '_labels.csv')

        # hand labels
        hand_labels_file = os.path.join(
            hparams['data_dir'], 'labels-hand', expt_id + '_labels.csv')

        # define data generator signals
        signals.append(['markers', 'labels_weak', 'labels_strong'])
        transforms.append([ZScore(), None, None])
        paths.append([markers_file, labels_file, hand_labels_file])

    # compute padding needed to account for convolutions
    hparams['batch_pad'] = compute_batch_pad(hparams)

    # build data generator
    print('Loading data...')
    data_gen = DataGenerator(
        hparams['expt_ids'], signals, transforms, paths, device=hparams['device'],
        batch_size=hparams['batch_size'], trial_splits=hparams['trial_splits'],
        train_frac=hparams['train_frac'], batch_pad=hparams['batch_pad'],
        input_type=hparams.get('input_type', 'markers'))
    print(data_gen)

    # automatically compute input/output sizes from data
    hparams['input_size'] = data_gen.datasets[0].data['markers'][0].shape[1]
    # try:
    #     hparams['output_size'] = data_gen.datasets[0].data['labels_strong'][0].shape[1]
    # except KeyError:
    #     hparams['output_size'] = data_gen.datasets[0].data['labels_weak'][0].shape[1]

    # -------------------------------------
    # extract data from generator as array
    # -------------------------------------
    data_dict, _ = get_data_by_dtype(data_gen, data_key='markers')
    data_mat_train = np.vstack(data_dict['train'])

    label_dict, _ = get_data_by_dtype(data_gen, data_key='labels_strong')
    label_mat_train = np.vstack(label_dict['train'])

    # -------------------------------------
    # build model
    # -------------------------------------
    hparams['rng_seed_model'] = hparams['rng_seed_train']
    np.random.seed(hparams['rng_seed_model'])
    model = RandomForestClassifier(
        n_estimators=2000, max_depth=6, max_features='auto', min_samples_leaf=5, bootstrap=True,
        n_jobs=-1, random_state=hparams['rng_seed_model'])
    print(model)

    # -------------------------------------
    # train model
    # -------------------------------------
    t_beg = time.time()
    # just select non-background points
    model.fit(data_mat_train[label_mat_train > 0, :], label_mat_train[label_mat_train > 0])
    t_end = time.time()
    print('Fit time: %.1f sec' % (t_end - t_beg))

    # save model
    with open(os.path.join(model_save_path, 'best_val_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams)

    # get rid of unneeded logging info
    clean_tt_dir(hparams)


def get_data_by_dtype(data_generator, sess_idxs=0, data_key='markers'):
    """Collect data from data generator and put into dictionary with dtypes for keys.

    TODO ------

    Parameters
    ----------
    data_generator : DataGenerator
    sess_idxs : int or list
        concatenate train/test/val data across one or more sessions
    data_key : str
        key into data generator object; 'markers' | 'labels_strong' | 'labels_weak'

    Returns
    -------
    tuple
        - data (dict): with keys 'train', 'val', 'test'
        - trial indices (dict): with keys 'train', 'val', 'test'

    """
    if isinstance(sess_idxs, int):
        sess_idxs = [sess_idxs]
    dtypes = ['train', 'val', 'test']
    data = {key: [] for key in dtypes}
    trial_idxs = {key: [] for key in dtypes}
    for sess_idx in sess_idxs:
        dataset = data_generator.datasets[sess_idx]
        for data_type in dtypes:
            curr_idxs = dataset.batch_idxs[data_type]
            trial_idxs[data_type] += list(curr_idxs)
            data[data_type] += [dataset[i_trial][data_key][0][:] for i_trial in curr_idxs]
    return data, trial_idxs


if __name__ == '__main__':

    """To run:

    (daart) $: python fit_models.py --data_config /path/to/data.yaml 
       --model_config /path/to/model.yaml --train_config /path/to/train.yaml

    For example yaml files, see the `configs` subdirectory inside the daart home directory

    """

    hyperparams = get_all_params()

    # if hyperparams.device == 'cuda':
    #     if isinstance(hyperparams.gpus_vis, int):
    #         gpu_ids = [str(hyperparams.gpus_vis)]
    #     else:
    #         gpu_ids = hyperparams.gpus_vis.split(';')
    #     hyperparams.optimize_parallel_gpu(
    #         run_main,
    #         gpu_ids=gpu_ids)
    #
    # elif hyperparams.device == 'cpu':
    hyperparams.optimize_parallel_cpu(
        run_main,
        nb_trials=hyperparams.tt_n_cpu_trials,
        nb_workers=hyperparams.tt_n_cpu_workers)
