"""Fit daart models (potentially searching hyperparameters on a set of data)."""

import logging
import numpy as np
import os
import sys
import time
import torch

from daart.data import DataGenerator, compute_sequence_pad
from daart.eval import plot_training_curves
from daart.io import export_hparams
from daart.models import Segmenter
from daart.testtube import get_all_params, print_hparams, create_tt_experiment, clean_tt_dir
from daart.train import Trainer
from daart.utils import build_data_generator


def run_main(hparams, *args):

    if not isinstance(hparams, dict):
        hparams = vars(hparams)

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

    # set up error logging (different from train logging)
    logging.basicConfig(
        filename=os.path.join(hparams['tt_version_dir'], 'console.log'),
        filemode='w', level=logging.DEBUG,
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # add logging to console
    logging.getLogger('matplotlib.font_manager').disabled = True

    # run train model script
    try:
        train_model(hparams)
    except:
        logging.exception('error_traceback')


def train_model(hparams):

    # print hparams to console
    print_str = print_hparams(hparams)
    logging.info(print_str)

    # take care of subsampled expt_ids
    if 'expt_ids_to_keep' not in hparams.keys():
        hparams['expt_ids_to_keep'] = hparams['expt_ids']
    else:
        hparams['expt_ids_to_keep'] = hparams['expt_ids_to_keep'].split(';')

    # where model results will be saved
    model_save_path = hparams['tt_version_dir']

    # build data generator
    data_gen = build_data_generator(hparams)
    logging.info(data_gen)

    # pull class weights out of labeled training data
    if hparams.get('weight_classes', False):
        pad = hparams['sequence_pad']
        totals = np.zeros((hparams['output_size'],))
        for dataset in data_gen.datasets:
            for b, batch in enumerate(dataset.data['labels_strong']):
                counts = np.bincount(batch[pad:-pad].astype('int'))
                if len(counts) == len(totals):
                    totals += counts
                else:
                    for i, c in enumerate(counts):
                        totals[i] += c
        totals[0] = 0  # get rid of background class
        # select class weights by choosing class with max labeled examples to have a value of 1;
        # the remaining weights will be inversely proportional to their prevalence. For example, a
        # class that has half as many examples as the most prevalent will be weighted twice as much
        class_weights = np.max(totals) / (totals + 1e-10)
        class_weights[totals == 0] = 0
        hparams['class_weights'] = [float(t) for t in class_weights]
        print('class weights: {}'.format(class_weights))
    else:
        hparams['class_weights'] = None

    # fit models
    if hparams['model_class'] == 'random-forest' or hparams['model_class'] == 'xgboost':

        import pickle
        from daart_utils.testtube import get_data_by_dtype

        # -------------------------------------
        # extract data from generator as array
        # -------------------------------------
        data_dict, _ = get_data_by_dtype(data_gen, data_key='markers', as_numpy=True)
        data_mat_train = np.vstack(data_dict['train'])
        # create time embedding of input data
        if hparams['input_type'] != 'features-simba':
            n_lags = hparams['n_lags']
            data_mat_train = np.hstack([
                np.roll(data_mat_train, i, axis=0) for i in range(-n_lags, n_lags + 1)])

        label_dict, _ = get_data_by_dtype(data_gen, data_key='labels_strong', as_numpy=True)
        label_mat_train = np.concatenate(label_dict['train'])

        # -------------------------------------
        # build model
        # -------------------------------------
        hparams['rng_seed_model'] = hparams['rng_seed_train']
        np.random.seed(hparams['rng_seed_model'])
        if hparams['model_class'] == 'random-forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=6000, max_features='sqrt', criterion='entropy', min_samples_leaf=1,
                bootstrap=True, n_jobs=-1, random_state=hparams['rng_seed_model'])
        elif hparams['model_class'] == 'xgboost':
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=2000, max_depth=3, learning_rate=0.1, objective='multi:softprob',
                eval_metric='mlogloss', tree_method='hist', gamma=1, min_child_weight=1,
                subsample=0.8, colsample_bytree=0.8, random_state=hparams['rng_seed_model'])
        logging.info(model)

        # -------------------------------------
        # train model
        # -------------------------------------
        t_beg = time.time()
        # just select non-background points
        model.fit(data_mat_train[label_mat_train > 0, :], label_mat_train[label_mat_train > 0])
        t_end = time.time()
        logging.info('Fit time: %.1f sec' % (t_end - t_beg))

        # save model
        save_file = os.path.join(model_save_path, 'best_val_model.pt')
        if hparams['model_class'] == 'random-forest':
            with open(save_file, 'wb') as f:
                pickle.dump(model, f)
        else:
            model.save_model(save_file)

    else:

        # -------------------------------------
        # build model
        # -------------------------------------
        hparams['rng_seed_model'] = hparams['rng_seed_train']  # TODO: get rid of this
        torch.manual_seed(hparams.get('rng_seed_model', 0))
        model = Segmenter(hparams)
        model.to(hparams['device'])
        logging.info(model)

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
                # set min_epochs to when annealings ends
                hparams['min_epochs'] = hparams['anneal_end']
        if hparams.get('variational', False):
            from daart.callbacks import AnnealHparam
            callbacks.append(AnnealHparam(
                hparams=model.hparams, key='kl_weight', epoch_start=0, epoch_end=100))

        # -------------------------------------
        # train model + cleanup
        # -------------------------------------
        t_beg = time.time()
        trainer = Trainer(**hparams, callbacks=callbacks)
        trainer.fit(model, data_gen, save_path=model_save_path)
        t_end = time.time()
        logging.info('Fit time: %.1f sec' % (t_end - t_beg))

        # save training curves
        if hparams.get('plot_train_curves', False):
            logging.info('Exporting train/val plots...')
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
