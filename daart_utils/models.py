"""Helper functions for managing models."""

import numpy as np
import os


def compute_model_predictions(hparams, data_gen, save_states=False, overwrite_states=False):
    from daart.io import find_experiment
    from daart.models import Segmenter

    version_int = find_experiment(hparams)
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
        arch_file = os.path.join(version_dir, 'hparams.pkl')
        # print('Loading model defined in %s' % arch_file)
        with open(arch_file, 'rb') as f:
            hparams_new = pickle.load(f)
        hparams_new['device'] = hparams.get('device', 'cpu')
        model = Segmenter(hparams_new)
        model.load_parameters_from_file(model_file)
        model.to(hparams_new['device'])
        model.eval()

        # compute predictions
        predictions = model.predict_labels(data_gen_test)['labels']
        predictions = np.argmax(np.vstack(predictions[0]), axis=1)

        # save predictions
        if save_states:
            print('saving states to %s' % states_file)
            np.save(states_file, predictions)

    return predictions
