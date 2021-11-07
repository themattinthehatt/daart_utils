"""Hard-coded session combinations for training/testing daart models on resident-intruder data."""

label_names = [
    'res_ear_left',
    'res_ear_right',
    'res_snout',
    'res_centroid',
    'res_left_flank',
    'res_right_flank',
    'res_tail_base',
    'res_tail_tip',
    'int_ear_left',
    'int_ear_right',
    'int_snout',
    'int_centroid',
    'int_left_flank',
    'int_right_flank',
    'int_tail_base',
    'int_tail_tip'
]

# test ids that are not used for testing
SESS_IDS_TEST = [
]

SESS_IDS_TRAIN_10 = [[  # choose sessions with most grooming (can be highly variable)

]]

SESS_IDS_ALL = [
    'exp24_mouse2_day2',
    'exp24_mouse2_day4',
    'exp24_mouse2_day5',
    'exp25_mouse1_day6',
    'exp25_mouse2_day2',
    'exp25_mouse2_day6',
    # w/o hand labels
    'exp24_mouse2_day1',
    'exp24_mouse2_day6',
    'exp25_mouse2_day3',
    'exp25_mouse2_day4',
    'exp32_mouse1_day1',
    'exp32_mouse2_day1',
    'exp33_mouse1_day4',
    'exp33_mouse1_day5',
]

# sessions with hand labels
SESS_IDS_LABELED = [
    'exp24_mouse2_day2',
    'exp24_mouse2_day4',
    'exp24_mouse2_day5',
    'exp25_mouse1_day6',
    'exp25_mouse2_day2',
    'exp25_mouse2_day6',
]

# sessions without hand labels
SESS_IDS_UNLABELED = [
    'exp24_mouse2_day1',
    'exp24_mouse2_day6',
    'exp25_mouse2_day3',
    'exp25_mouse2_day4',
    'exp32_mouse1_day1',
    'exp32_mouse2_day1',
    'exp33_mouse1_day4',
    'exp33_mouse1_day5',
]
