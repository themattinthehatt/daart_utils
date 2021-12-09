"""Helper functions for managing models."""

import numpy as np
import os
import pickle
import yaml


def compute_model_predictions(
        hparams, data_gen, verbose=False, save_states=False, overwrite_states=False):
    """Compute predictions given hparams dict and data generator. Optionally save/overwrite results.

    Parameters
    ----------
    hparams : dict
        contains keys for all parameters necessary to define a model; see
        :func:daart_utils.models.get_default_hparams for an example.
    data_gen : daart.data.DataGenerator object
        data generator that serves markers to predict labels from
    verbose: bool
        print out results from `find_experiment`
    save_states : bool, optional
        True to save predictions in model directory
    overwrite_states : bool, optional
        if prediction files are already present, function will automatically load this file unless
        `overwrite` is True

    Returns
    -------
    np.array
        argmax of model softmax outputs

    """
    from daart.io import find_experiment
    from daart.models import Segmenter

    version_int = find_experiment(hparams, verbose=verbose)
    if version_int is None:
        raise FileNotFoundError
    version_str = str('version_%i' % version_int)
    version_dir = os.path.join(hparams['tt_expt_dir'], version_str)

    # check to see if states exist
    states_file = os.path.join(version_dir, '%s_states.npy' % data_gen.datasets[0].id)
    if os.path.exists(states_file) and not overwrite_states:
        predictions = np.load(states_file)
    else:
        # load model
        model_file = os.path.join(version_dir, 'best_val_model.pt')
        try:
            arch_file = os.path.join(version_dir, 'hparams.pkl')
            # print('Loading model defined in %s' % arch_file)
            with open(arch_file, 'rb') as f:
                hparams_new = pickle.load(f)
        except FileNotFoundError:
            arch_file = os.path.join(version_dir, 'hparams.yaml')
            with open(arch_file, 'r') as f:
                hparams_new = yaml.safe_load(f)

        hparams_new['device'] = hparams.get('device', 'cpu')

        if hparams_new['model_type'] == 'random-forest':
            with open(os.path.join(model_file), 'rb') as f:
                model = pickle.load(f)
            predictions_ = predict_labels_with_rf(model, data_gen)
            predictions = predictions_[0]  # assume a single session
        else:
            model = Segmenter(hparams_new)
            model.load_parameters_from_file(model_file)
            model.to(hparams_new['device'])
            model.eval()
            predictions = model.predict_labels(data_gen)['labels']
            predictions = np.argmax(np.vstack(predictions[0]), axis=1)

        # save predictions
        if save_states:
            print('saving states to %s' % states_file)
            np.save(states_file, predictions)

    return predictions


def get_default_hparams(**kwargs):
    """Return default hparam dict that specifies uniquely identifying data and model parameters.

    Parameters
    ----------
    kwargs
        use the kwargs to overwrite any defaults; see Examples below

    Examples
    --------
    hparams = get_default_hparams(n_lags=8, model_type='gru', device='gpu')

    Returns
    -------
    dict

    """
    hparams = {
        'rng_seed_train': 0,        # rng seed for batch order
        'rng_seed_model': 0,        # rng seed for model initialization
        'trial_splits': '9;1;0;0',  # 'train;val;test;gap'
        'train_frac': 1,            # fraction of initial training data to use
        'batch_size': 2000,         # batch size
        'model_type': 'dtcn',       # 'temporal-mlp' | 'dtcn' | 'gru' | 'lstm'
        'learning_rate': 1e-4,      # adam learning rate
        'n_hid_layers': 2,          # hidden layers for each of the encoder/decoder networks
        'n_hid_units': 32,          # hidden units for each hidden layer
        'n_lags': 4,                # lags in convolution filters (t - n_lags, t + n_lags)
        'l2_reg': 0,                # l2 reg on model parameters
        'lambda_strong': 1,         # loss weight on hand label classification
        'lambda_weak': 1,           # loss weight on heuristic label classification
        'lambda_pred': 1,           # loss weight on next-step-ahead prediction
        'bidirectional': True,      # for 'gru' and 'lstm' models
        'device': 'cpu',            # computational device on which to place model/data
        'dropout': 0.1,             # dropout for 'dtcn'
        'activation': 'lrelu',      # hidden unit activation function
        'batch_pad': 0,             # batch pad for convolutions; needs to be updated for non-rnns
        'tt_expt_dir': 'test',      # test-tube experiment directory name
    }
    # update hparams with user-provided kwargs
    for key, val in kwargs.items():
        hparams[key] = val
    return hparams


def predict_labels_with_rf(model, data_generator):
    """

    Parameters
    ----------
    model : sklearn.embedding.RandomForestClassifier object
        trained model
    data_generator : DataGenerator object
        data generator to serve data batches

    Returns
    -------
    list of lists
        first list is over datasets; second list is over batches in the dataset; each element is a
        numpy array of the predicted class

    """

    # initialize container for inputs
    inputs = [[] for _ in range(data_generator.n_datasets)]
    for sess, dataset in enumerate(data_generator.datasets):
        inputs[sess] = [np.array([]) for _ in range(dataset.n_trials)]

    # partially fill container (gap trials will be included as nans)
    dtypes = ['train', 'val', 'test']
    for dtype in dtypes:
        data_generator.reset_iterators(dtype)
        for i in range(data_generator.n_tot_batches[dtype]):
            data, sess = data_generator.next_batch(dtype)
            inputs[sess][data['batch_idx'].item()] = data['markers'][0].cpu().numpy()

    labels = [model.predict(np.vstack(ins)) for ins in inputs]

    return labels
