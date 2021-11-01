# daart_utils: utilities for training and analyzing daart models

## Installation

First you'll have to install the `daart` package, which contains the base modeling code - follow the 
directions [here](https://github.com/themattinthehatt/daart).
Then, in the command line, navigate to where you'd like to install the `daart_utils` package and move 
into that directory:
```
$: git clone https://github.com/themattinthehatt/daart_utils
$: cd daart_utils
```

Next, active the `daart` conda environment and locally install the `daart_utils` package.

```
$: conda activate daart
(daart) $: pip install -r requirements.txt
(daart) $: pip install -e .
```

## Set paths

To set user-specific paths that the scripts and notebooks can read from, create a file named
`daart_utils/daart_utils/paths.py` that looks like the following:

```python

# where daart config files are stored, i.e. `data.yaml`, `model.yaml`, and `train.yaml`
config_path = '/path/to/configs'

# data path
# ---------
# for example, hand labels for a particular session are located at
#
# `data_path/<dataset>/hand-labels/<session_id>_labels.csv`
#
# where <dataset> is `ibl`, `fly`, etc.
data_path = '/path/to/data'

# results path
# ------------
# for example, with the test-tube directory name 'grid-search' and a dtcn model,
# model info will be stored in the directory
#
# `base_path/<dataset>/<session-id>/dtcn/grid-search/version_X`
#
# where <dataset> is `ibl`, `fly`, etc.
results_path = '/path/to/results'

```

The scripts and notebooks will automatically replace the paths in the config files with the paths
defined in this `paths.py` file.


## Fit models

This package uses [test-tube](https://williamfalcon.github.io/test-tube/) for hyperparameter 
searching and model fitting. The script `scripts/fit_models_loop.py` will fit one or more models 
based on three yaml configuration files: one describing the data, one describing the model, and one 
describing the training procedure.
 
First copy/paste templates of the three config files from the `daart` directory into the location
defined by `config_path` above (`data.yaml`, `model.yaml`, and `train.yaml`). The default fitting 
script will expect dataset-specific data configs, so you should make copies of the above
files named `data_fly.yaml`, for example.

Next, configure the models you want to fit in the yaml files. The provided script will 
automatically hyperparameter search over lists, so if you set `lambda_weak: [0, 1]` in the model yaml
then two models will be fit; if you additionally set `lambda_pred: [0, 1]`, then all four model
combinations will be fit.

Once you have set the desired parameters in the config files, you can run the fitting script from 
the command line by providing a dataset and a model type; for example, to fit a dTCN model on fly 
data, run the following:

```
(daart) $: python scripts/fit_models_loop.py --dataset fly --fit_dtcn
```

See the script for more details on options.

You can then explore the results with the jupyter notebook 
`notebooks/evaluation_across_models.ipynb`
