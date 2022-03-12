"""Hard-coded session combinations for training/testing daart models on resident-intruder data."""

label_names = [
    'background', 'attack', 'anogenital_sniffing', 'mounting', 'allogrooming_normal',
    'allogroom_vigorous', 'intruder_sniff'
]

# test ids that are not used for testing
SESS_IDS_TEST = [
    'exp24_mouse2_day2',
    'exp25_mouse1_day6',
    'exp25_mouse2_day2',
]

SESS_IDS_TRAIN_9 = [[
    # w/ hand labels
    'exp24_mouse2_day4',
    'exp24_mouse2_day5',
    'exp25_mouse2_day6',
    # w/o hand labels
    'exp24_mouse2_day1',
    'exp25_mouse2_day4',
    'exp32_mouse1_day1',
    'exp32_mouse2_day1',
    'exp33_mouse1_day4',
    'exp33_mouse1_day5',
]]

SESS_IDS_ALL = [
    # w/ hand labels
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
