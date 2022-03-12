"""Hard-coded session combinations for training/testing daart models on resident-intruder data."""

label_names = [
    'background', 'other', 'face_grooming', 'body_grooming', 'anogenital_sniffing', 'mounting',
    'vigorous_grooming', 'intruder_sniffing', 'attack',
]

# test ids that are not used for testing
SESS_IDS_TEST = [
    'day1_mouse2',
    'day2_mouse3',
    'day4_mouse1',
    'day5_mouse2',
]

SESS_IDS_TRAIN_6 = [[
    'day2_mouse2.2',
    'day2_mouse2',
    'day3_mouse1',
    'day3_mouse3',
    'day4_mouse2',
    'day6_mouse5',
]]

SESS_IDS_ALL = [
    'day1_mouse2',
    'day2_mouse2.2',
    'day2_mouse2',
    'day2_mouse3',
    'day3_mouse1',
    'day3_mouse3',
    'day4_mouse1',
    'day4_mouse2',
    'day5_mouse2',
    'day6_mouse5',
]

# sessions with hand labels
SESS_IDS_LABELED = [
]

# sessions without hand labels
SESS_IDS_UNLABELED = [
]
