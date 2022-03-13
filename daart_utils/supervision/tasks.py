"""Compute tasks from pose tracking outputs."""

import numpy as np
import pandas as pd


def fly(markers, ball_me):
    """Compute tasks to help classify fly behaviors.

    NOTE: needs access to external data (ball motion energy)
    TODO: Dropbox data not complete enough to recreate tasks for fly data

    Parameters
    ----------
    markers : dict
        output of pose estimation; key is bodypart, val is np.ndarray of shape (n_t, 2)
    ball_me : array-like
        ball motion energy

    Returns
    -------
    pd.DataFrame

    """

    from skimage.restoration import denoise_tv_chambolle

    # compute motion energy for each set of limbs and body; clip large values likely due to noise
    ms_fg = np.hstack([markers['fore-top'], markers['fore-bot']])
    me_fg = np.concatenate([np.zeros((1, 4)), np.square(np.diff(ms_fg, axis=0))], axis=0)
    xs_fg = np.mean(me_fg, axis=1)
    xs_fg = np.clip(xs_fg, a_min=None, a_max=np.percentile(xs_fg, 99))
    xs_fg_d = denoise_tv_chambolle(xs_fg, weight=0.05)

    ms_mg = np.hstack([markers['mid-top'], markers['mid-bot']])
    me_mg = np.concatenate([np.zeros((1, 4)), np.square(np.diff(ms_mg, axis=0))], axis=0)
    xs_mg = np.mean(me_mg, axis=1)
    xs_mg = np.clip(xs_mg, a_min=None, a_max=np.percentile(xs_mg, 99))
    xs_mg_d = denoise_tv_chambolle(xs_mg, weight=0.05)

    ms_bg = np.hstack([markers['hind-top'], markers['hind-bot']])
    me_bg = np.concatenate([np.zeros((1, 4)), np.square(np.diff(ms_bg, axis=0))], axis=0)
    xs_bg = np.mean(me_bg, axis=1)
    xs_bg = np.clip(xs_bg, a_min=None, a_max=np.percentile(xs_bg, 99))
    xs_bg_d = denoise_tv_chambolle(xs_bg, weight=0.05)

    ms_ab = np.hstack([markers['abdomen-top'], markers['addomen-bot']])
    me_ab = np.concatenate([np.zeros((1, 4)), np.square(np.diff(ms_ab, axis=0))], axis=0)
    xs_ab = np.mean(me_ab, axis=1)
    xs_ab = np.clip(xs_ab, a_min=None, a_max=np.percentile(xs_ab, 99))
    xs_ab_d = denoise_tv_chambolle(xs_ab, weight=0.05)

    t = ball_me.shape[0]

    df = pd.DataFrame(
        np.column_stack([ball_me[:t], xs_fg_d[:t], xs_mg_d[:t], xs_bg_d[:t], xs_ab_d[:t]]),
        columns=['ball_me', 'forelimb_me', 'midlimb_me', 'hindlimb_me', 'abdomen_me']
    )

    return df


def ibl(markers, markers_nose, wheel_vel):
    """Compute tasks to help classify ibl behaviors.

    NOTE: needs access to external data (wheel velocity, nose markers)
    TODO: Dropbox data not complete enough to recreate tasks for ibl data

    Parameters
    ----------
    markers : np.ndarray
    markers_nose : np.ndarray
    wheel_vel : array-like

    Returns
    -------
    pd.DataFrame

    """

    from skimage.restoration import denoise_tv_chambolle
    from ibl_utils.markers import compute_distance

    me = np.concatenate([
        np.zeros((1, markers.shape[1])),
        np.square(np.diff(markers, axis=0))
    ], axis=0)

    dist_paw_nose = compute_distance(markers, np.nanmedian(markers_nose, axis=0))

    paw_me = np.mean(me, axis=1)
    paw_me = np.clip(paw_me, a_min=None, a_max=np.percentile(paw_me, 99))
    paw_me_d = denoise_tv_chambolle(paw_me, weight=0.05)

    # turning - paw me and wheel velocity above thresholds, paw far away from nose
    wh_speed = np.abs(wheel_vel)

    df = pd.DataFrame(
        np.column_stack([dist_paw_nose, paw_me_d, wh_speed]),
        columns=['dist_paw_nose', 'paw_me', 'wheel_speed']
    )

    return df


def oft(markers, pixels):
    """Compute tasks to help classify oft behaviors.

    NOTE: needs to be run on raw markers (which includes arena corners), not aligned markers

    Parameters
    ----------
    markers : np.ndarray
    pixels : np.ndarray
        shape (T, 2), first column is n_pixels mouse, second column is n_pixels shadow

    Returns
    -------
    dict

    """
    from daart_utils.utils import point_to_line_distance, point_to_point_distance

    # compute distance of mouse to boundary
    bl = np.median(markers['bl'], axis=0)
    tl = np.median(markers['tl'], axis=0)
    br = np.median(markers['br'], axis=0)
    tr = np.median(markers['tr'], axis=0)
    mouse_to_l_bound = np.abs(point_to_line_distance(markers['bodycentre'], tl, bl))
    mouse_to_r_bound = np.abs(point_to_line_distance(markers['bodycentre'], tr, br))
    mouse_to_t_bound = np.abs(point_to_line_distance(markers['bodycentre'], tr, tl))
    mouse_to_b_bound = np.abs(point_to_line_distance(markers['bodycentre'], br, bl))
    mouse_to_boundary = np.min(np.hstack([
        mouse_to_l_bound[:, None], mouse_to_r_bound[:, None],
        mouse_to_t_bound[:, None], mouse_to_b_bound[:, None]]), axis=1)

    # neck_to_body = point_to_point_distance(markers['bodycentre'], markers['neck'])
    # head_to_neck = point_to_point_distance(markers['headcentre'], markers['neck'])
    # nose_to_head = point_to_point_distance(markers['nose'], markers['headcentre'])
    # nose_to_neck = point_to_point_distance(markers['nose'], markers['neck'])
    nose_to_tail = point_to_point_distance(markers['nose'], markers['tailbase'])
    speed = np.concatenate([
        [0],
        point_to_point_distance(markers['bodycentre'][:-1], markers['bodycentre'][1:])])

    body_pixels = pixels[:, 0]

    shadow_body_ratio = pixels[:, 1] / pixels[:, 0]
    shadow_body_ratio[pixels[:, 0] == 0] = 1

    height_width_ratio = pixels[:, 2] / pixels[:, 3]
    height_width_ratio[pixels[:, 3] == 0] = 1

    data = np.column_stack([
        mouse_to_boundary,
        nose_to_tail,
        speed,
        body_pixels,
        shadow_body_ratio,
        height_width_ratio
    ])
    column_names = [
        'mouse_to_boundary',
        'nose_to_tail',
        'speed',
        'body_pixels',
        'shadow_to_body_ratio',
        'height_to_width_ratio'
    ]

    df = pd.DataFrame(data, columns=column_names)

    return df


def resident_intruder_old(features, feature_names):
    """Compute heuristics for resident-intruder behaviors.

    NOTE: needs to be run on simba features, not markers

    Parameters
    ----------
    features : np.ndarray
    feature_names : np.ndarray

    Returns
    -------
    pd.DataFrame

    """

    column_names = [
        'Centroid_distance',  # distance between the centroid of the CD1 and centroid of the C57
        'M1_Nose_to_M2_lat_left',  # distance between the nose of the CD1 and lateral left side of the C57
        'M1_Nose_to_M2_lat_right',  # distance between the nose of the CD1 and lateral right side of the C57
        'M2_Nose_to_M1_lat_left',  # distance between the nose of the C57 and lateral left side of the CD1
        'M2_Nose_to_M1_lat_right',  # distance between the nose of the C57 and lateral right side of the CD1
        'M1_Nose_to_M2_tail_base',  # distance between the nose of the CD1 and tail-base of the C57
        'M2_Nose_to_M1_tail_base',  # distance between the nose of the C57 and tail-base of the CD1
        'Total_movement_centroids',  # The sum of movement of the CD1 centroid and C57 centroid from previous frame
    ]

    data = []
    for col_name in column_names:
        idx_feature = np.where(feature_names == col_name)[0][0]
        data.append(features[:, idx_feature])

    column_names += ['Angle_between_M1_M2']
    idx1 = np.where(feature_names == 'Mouse_1_angle')[0][0]
    idx2 = np.where(feature_names == 'Mouse_2_angle')[0][0]
    data.append(features[:, idx1] - features[:, idx2])

    df = pd.DataFrame(np.column_stack(data), columns=column_names)

    return df


def resident_intruder(features, feature_names):
    """Compute heuristics for resident-intruder behaviors.

    NOTE: needs to be run on simba features, not markers

    Parameters
    ----------
    features : np.ndarray
    feature_names : np.ndarray

    Returns
    -------
    pd.DataFrame

    """

    column_names = [
        'Centroid_distance',  # distance between the centroid of the CD1 and centroid of the C57
        'M1_Nose_to_M2_tail_base',  # distance between the nose of the CD1 and tail-base of the C57
        'M1_Nose_to_M2_lat_left',  # distance between the nose of the CD1 and lateral left side of the C57
        'M1_Nose_to_M2_lat_right',  # distance between the nose of the CD1 and lateral right side of the C57
        'M2_Nose_to_M1_tail_base',  # distance between the nose of the C57 and tail-base of the CD1
        'M2_Nose_to_M1_lat_left',  # distance between the nose of the C57 and lateral left side of the CD1
        'M2_Nose_to_M1_lat_right',  # distance between the nose of the C57 and lateral right side of the CD1
        'Total_movement_centroids',  # The sum of movement of the CD1 centroid and C57 centroid from previous frame
        'Total_movement_all_bodyparts_both_mice_deviation',
        'Nose_movement_M1_median_15',
    ]

    data = []
    for col_name in column_names:
        idx_feature = np.where(feature_names == col_name)[0][0]
        data.append(features[:, idx_feature])

    # nose movement relative to centroid for mouse 1
    column_names += ['Mouse_1_nose_movement_relative_to_centroid']
    idx_feature = np.where(feature_names == 'Mouse_1_Nose_to_centroid')[0][0]
    tmp = features[:, idx_feature]
    data.append(np.abs(np.concatenate([[0], np.diff(tmp)])))

    # nose movement relative to centroid for mouse 2
    column_names += ['Mouse_2_nose_movement_relative_to_centroid']
    idx_feature = np.where(feature_names == 'Mouse_2_Nose_to_centroid')[0][0]
    tmp = features[:, idx_feature]
    data.append(np.abs(np.concatenate([[0], np.diff(tmp)])))

    df = pd.DataFrame(np.column_stack(data), columns=column_names)

    return df


def calms21(markers, features, feature_names):
    """Compute heuristics for resident-intruder behaviors of CalMS21 dataset.

    Replicates the programs defined in https://arxiv.org/pdf/2011.13917.pdf.

    NOTE: needs to be run on simba features AND markers

    Parameters
    ----------
    markers : dict
    features : np.ndarray
    feature_names : np.ndarray

    Returns
    -------
    pd.DataFrame

    """
    from daart_utils.utils import absolute_angle

    column_names = [
        'Movement_mouse_1_centroid',  # movements of CD1 centroid from previous frame
        'Movement_mouse_2_centroid',  # movements of C57 centroid from previous frame
        'Nose_to_nose_distance',  # distance between the nose of the CD1 and nose of the C57
        'M1_Nose_to_M2_tail_base',  # distance between the nose of the CD1 and tail-base of the C57
        'Mouse_1_angle',  # The body angle in degrees (using the CD1 tail-base, centroid, and nose coordinates)
        'Mouse_2_angle',  # The body angle in degrees (using the C57 tail-base, centroid, and nose coordinates)
        'Centroid_distance',  # distance between the centroid of the CD1 and centroid of the C57
    ]

    data = []
    for col_name in column_names:
        idx_feature = np.where(feature_names == col_name)[0][0]
        data.append(features[:, idx_feature])

    # facing angle
    column_names += ['Facing_angle']
    feat_facing_angle = absolute_angle(markers['tailbase_2'], markers['neck_1'], markers['neck_2'])
    data.append(feat_facing_angle)

    # nose movement relative to centroid for mouse 1
    column_names += ['Mouse_1_nose_movement_relative_to_centroid']
    idx_feature = np.where(feature_names == 'Mouse_1_Nose_to_centroid')[0][0]
    tmp = features[:, idx_feature]
    data.append(np.abs(np.concatenate([[0], np.diff(tmp)])))

    # nose movement relative to centroid for mouse 2
    column_names += ['Mouse_2_nose_movement_relative_to_centroid']
    idx_feature = np.where(feature_names == 'Mouse_2_Nose_to_centroid')[0][0]
    tmp = features[:, idx_feature]
    data.append(np.abs(np.concatenate([[0], np.diff(tmp)])))

    df = pd.DataFrame(np.column_stack(data), columns=column_names)

    return df
