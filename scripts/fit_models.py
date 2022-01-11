"""Fit daart models (potentially searching hyperparameters on a set of data)."""

import numpy as np
import os
import time
import torch

from daart.data import DataGenerator
from daart.eval import plot_training_curves
from daart.io import export_hparams
from daart.models import Segmenter
from daart.train import Trainer
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

    # take care of subsampled expt_ids
    if 'expt_ids_to_keep' not in hparams.keys():
        hparams['expt_ids_to_keep'] = hparams['expt_ids']
    else:
        hparams['expt_ids_to_keep'] = hparams['expt_ids_to_keep'].split(';')

    # where model results will be saved
    model_save_path = hparams['tt_version_dir']

    # -------------------------------------
    # build data generator
    # -------------------------------------
    signals = []
    transforms = []
    paths = []

    for expt_id in hparams['expt_ids']:

        signals_curr = []
        transforms_curr = []
        paths_curr = []

        # DLC markers or features (i.e. from simba)
        input_type = hparams.get('input_type', 'markers')
        markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.h5')
        if not os.path.exists(markers_file):
            markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.csv')
        if not os.path.exists(markers_file):
            markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.npy')
        if not os.path.exists(markers_file):
            raise FileNotFoundError('could not find marker file for %s' % expt_id)
        signals_curr.append('markers')
        transforms_curr.append(ZScore())
        paths_curr.append(markers_file)

        # hand labels
        if hparams.get('lambda_strong', 0) > 0:
            if expt_id not in hparams['expt_ids_to_keep']:
                hand_labels_file = None
            else:
                hand_labels_file = os.path.join(
                    hparams['data_dir'], 'labels-hand', expt_id + '_labels.csv')
                if not os.path.exists(hand_labels_file):
                    hand_labels_file = None
            signals_curr.append('labels_strong')
            transforms_curr.append(None)
            paths_curr.append(hand_labels_file)

        # heuristic labels
        if hparams.get('lambda_weak', 0) > 0:
            heur_labels_file = os.path.join(
                hparams['data_dir'], 'labels-heuristic', expt_id + '_labels.csv')
            signals_curr.append('labels_weak')
            transforms_curr.append(None)
            paths_curr.append(heur_labels_file)

        # tasks
        if hparams.get('lambda_task', 0) > 0:
            tasks_labels_file = os.path.join(hparams['data_dir'], 'tasks', expt_id + '.csv')
            signals_curr.append('tasks')
            transforms_curr.append(ZScore())
            paths_curr.append(tasks_labels_file)

        # define data generator signals
        signals.append(signals_curr)
        transforms.append(transforms_curr)
        paths.append(paths_curr)

    # compute padding needed to account for convolutions
    if hparams['model_type'] == 'random-forest':
        hparams['batch_pad'] = 0
    else:
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
    if hparams.get('lambda_task', 0) > 0:
        task_size = 0
        for batch in data_gen.datasets[0].data['tasks']:
            if batch.shape[1] == 0:
                continue
            else:
                task_size = batch.shape[1]
                break
        hparams['task_size'] = task_size

    if hparams['model_type'] == 'random-forest':

        import pickle
        from sklearn.ensemble import RandomForestClassifier
        from daart_utils.testtube import get_data_by_dtype

        # -------------------------------------
        # extract data from generator as array
        # -------------------------------------
        data_dict, _ = get_data_by_dtype(data_gen, data_key='markers', as_numpy=True)
        data_mat_train = np.vstack(data_dict['train'])

        label_dict, _ = get_data_by_dtype(data_gen, data_key='labels_strong', as_numpy=True)
        label_mat_train = np.concatenate(label_dict['train'])

        # -------------------------------------
        # build model
        # -------------------------------------
        hparams['rng_seed_model'] = hparams['rng_seed_train']
        np.random.seed(hparams['rng_seed_model'])
        model = RandomForestClassifier(
            n_estimators=6000, max_features='sqrt', criterion='entropy', min_samples_leaf=1,
            bootstrap=True, n_jobs=-1, random_state=hparams['rng_seed_model'])
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
        with open(os.path.join(model_save_path, 'best_val_model.pt'), 'wb') as f:
            pickle.dump(model, f)

    else:

        # -------------------------------------
        # build model
        # -------------------------------------
        hparams['rng_seed_model'] = hparams['rng_seed_train']  # TODO: get rid of this
        torch.manual_seed(hparams.get('rng_seed_model', 0))
        model = Segmenter(hparams)
        model.to(hparams['device'])
        print(model)

        # -------------------------------------
        # set up training callbacks
        # -------------------------------------
        callbacks = []
        if hparams['enable_early_stop']:
            from daart.callbacks import EarlyStopping
            # Note that patience does not account for val check interval values greater than 1;
            # for example, if val_check_interval=5 and patience=20, then the model will train
            # for at least 5 * 20 = 100 epochs before training can terminate
            callbacks.append(EarlyStopping(patience=hparams['early_stop_history']))
        if hparams.get('semi_supervised_algo', 'none') == 'pseudo_labels':
            from daart.callbacks import AnnealHparam, PseudoLabels
            if model.hparams['lambda_weak'] == 0:
                print('warning! use lambda_weak in model.yaml to weight pseudo label loss')
            else:
                callbacks.append(AnnealHparam(
                    hparams=model.hparams, key='lambda_weak', epoch_start=hparams['anneal_start'],
                    epoch_end=hparams['anneal_end']))
                callbacks.append(PseudoLabels(
                    prob_threshold=hparams['prob_threshold'], epoch_start=hparams['anneal_start']))

        # -------------------------------------
        # train model + cleanup
        # -------------------------------------
        t_beg = time.time()
        trainer = Trainer(**hparams, callbacks=callbacks)
        trainer.fit(model, data_gen, save_path=model_save_path)
        t_end = time.time()
        print('Fit time: %.1f sec' % (t_end - t_beg))

        # save training curves
        if hparams.get('plot_train_curves', False):
            print('\nExporting train/val plots...')
            hparam_str = ''
            if hparams.get('lambda_strong', 0) > 0:
                hparam_str += 'strong=%.1f' % hparams['lambda_strong']
            if hparams.get('lambda_weak', 0) > 0:
                hparam_str += '_weak=%.1f' % hparams['lambda_weak']
            if hparams.get('lambda_task', 0) > 0:
                hparam_str += '_task=%.1f' % hparams['lambda_task']
            if hparams.get('lambda_pred', 0) > 0:
                hparam_str += '_pred=%.1f' % hparams['lambda_pred']
            plot_training_curves(
                os.path.join(model_save_path, 'metrics.csv'), dtype='train',
                expt_ids=hparams['expt_ids'],
                save_file=os.path.join(model_save_path, 'train_curves_%s' % hparam_str),
                format='png')
            plot_training_curves(
                os.path.join(model_save_path, 'metrics.csv'), dtype='val',
                expt_ids=hparams['expt_ids'],
                save_file=os.path.join(model_save_path, 'val_curves_%s' % hparam_str),
                format='png')

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams)

    # get rid of unneeded logging info
    clean_tt_dir(hparams)


if __name__ == '__main__':

    """To run:

    (daart) $: python fit_models.py --data_config /path/to/data.yaml 
       --model_config /path/to/model.yaml --train_config /path/to/train.yaml

    For example yaml files, see the `configs` subdirectory inside the daart home directory

    """

    hyperparams = get_all_params()

    if hyperparams.device == 'cuda':
        if isinstance(hyperparams.gpus_vis, int):
            gpu_ids = [str(hyperparams.gpus_vis)]
        else:
            gpu_ids = hyperparams.gpus_vis.split(';')
        hyperparams.optimize_parallel_gpu(
            run_main,
            gpu_ids=gpu_ids)

    elif hyperparams.device == 'cpu':
        hyperparams.optimize_parallel_cpu(
            run_main,
            nb_trials=hyperparams.tt_n_cpu_trials,
            nb_workers=hyperparams.tt_n_cpu_workers)
