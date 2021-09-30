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
config_path = '/home/mattw/.daart'

# base path; for example, hand labels for a particular session are located at
#
# `base_path/[dataset]/hand-labels/[session_id]_labels.csv`
#
# where [dataset] is `ibl`, `fly`, etc.
base_path = '/home/mattw/Dropbox/shared/segmentation-data'

```
