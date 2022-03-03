"""Hard-coded session combinations for training/testing daart models on ibl data."""

label_names = ['background', 'still', 'move', 'wheel_turn', 'groom']

# test ids that are not used for testing
SESS_IDS_TEST = [
    'churchlandlab_CSHL045_2020-02-27-001',
    'cortexlab_KS020_2020-02-06-001',
    'hoferlab_SWC_043_2020-09-15-001',
    'mrsicflogellab_SWC_052_2020-10-22-001',
    'wittenlab_ibl_witten_27_2021-01-21-001',
]

SESS_IDS_TRAIN_5 = [[
    'danlab_DY_009_2020-02-27-001',
    'danlab_DY_018_2020-10-15-001',
    'hoferlab_SWC_061_2020-11-23-001',
    'mrsicflogellab_SWC_058_2020-12-11-001',
    'wittenlab_ibl_witten_26_2021-01-27-002',
]]

# SESS_IDS_TEST = [
#     'churchlandlab_CSHL045_2020-02-27-001',
#     'cortexlab_KS020_2020-02-06-001',
#     'danlab_DY_018_2020-10-15-001',
#     'hoferlab_SWC_043_2020-09-15-001',
#     'mrsicflogellab_SWC_052_2020-10-22-001',
#     'wittenlab_ibl_witten_27_2021-01-21-001',
#     'wittenlab_ibl_witten_26_2021-01-27-002',
# ]
#
# SESS_IDS_TRAIN_5 = [[
#     'danlab_DY_009_2020-02-27-001',
#     'hoferlab_SWC_061_2020-11-23-001',
#     'mrsicflogellab_SWC_058_2020-12-11-001',
# ]]

SESS_IDS_ALL = [
    'churchlandlab_CSHL045_2020-02-27-001',
    'cortexlab_KS020_2020-02-06-001',
    'danlab_DY_009_2020-02-27-001',
    'danlab_DY_018_2020-10-15-001',
    'hoferlab_SWC_043_2020-09-15-001',
    'hoferlab_SWC_061_2020-11-23-001',
    'mrsicflogellab_SWC_052_2020-10-22-001',
    'mrsicflogellab_SWC_058_2020-12-11-001',
    'wittenlab_ibl_witten_26_2021-01-27-002',
    'wittenlab_ibl_witten_27_2021-01-21-001',
]
