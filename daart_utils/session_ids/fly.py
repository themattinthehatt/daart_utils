"""Hard-coded session combinations for training/testing daart models on fly data."""

label_names = [
    'background', 'still', 'walk', 'front_groom', 'back_groom', 'abdomen-move', 'fidget']

# test ids that are not used for testing
SESS_IDS_TEST = [
    '2019_06_26_fly2',
    '2019_08_14_fly1',
    '2019_08_20_fly3',
    '2019_10_14_fly2',
    '2019_10_21_fly1',
]

# 5 ids for training
SESS_IDS_TRAIN_5 = [[
    '2019_08_07_fly2',
    '2019_08_08_fly1',
    '2019_08_20_fly2',
    '2019_10_10_fly3',
    '2019_10_14_fly3',
]]

# 10 ids for training
SESS_IDS_ALL = [
    '2019_08_07_fly2',
    '2019_08_08_fly1',
    '2019_08_20_fly2',
    '2019_10_10_fly3',
    '2019_10_14_fly3',
    '2019_06_26_fly2',
    '2019_08_14_fly1',
    '2019_08_20_fly3',
    '2019_10_14_fly2',
    '2019_10_21_fly1',
]
