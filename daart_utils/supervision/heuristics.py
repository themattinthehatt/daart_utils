"""Compute heuristic labels from pose tracking outputs."""

import numpy as np

from daart_utils.utils import get_label_runs


def fly(
        markers, ball_me, walk_thresh=0.5, still_thresh=0.05, groom_thresh=0.02, ab_thresh=0.5,
        fidget_thresh=0.1, run_smoother=True, n_restarts=3):
    """Compute heuristics for 6 fly behaviors: still, walk, front/back groom, ab raise, fidget.

    NOTE: needs access to external data (ball motion energy)
    TODO: Dropbox data not complete enough to recreate heuristic labels for fly data

    Parameters
    ----------
    markers : np.ndarray
        output of pose estimation; shape is (n_t, n_keypoints * 2)
    ball_me : array-like
        ball motion energy
    walk_thresh : float, optional
    still_thresh : float, optional
    groom_thresh : float, optional
    ab_thresh : float, optional
    fidget_thresh : float, optional
    run_smoother : bool, optional
    n_restarts : int, optional

    Returns
    -------
    dict

    """

    # failure modes:
    # still:
    #   - slow abdominal/grooming movements counted here; mvmt is slow but consistent over frames
    #   - doesn't work well if fly is almost always moving (06_26_fly1)
    # moving:
    #   - when using ball, will also pick up periods when ball is being pushed away (struggle)
    # front groom:
    #   - often finds walking where mid/back legs are not in motion, but front are (10_14)fly3)
    # back groom:
    #   - sometimes mid legs do back grooming, counted as moving/misc (08_08_fly1)
    # abdomen-move:
    #   - false positives when fly doesn't actually perform this behavior
    # fidget:
    #   - ?
    # misc:

    from skimage.restoration import denoise_tv_chambolle

    D = markers.shape[1]
    n_markers = int(D / 2)

    # idxs = {
    #     'body': np.array([0, 1, 0 + n_markers, 1 + n_markers]).astype('int'),
    #     'back': np.array([2, 3, 2 + n_markers, 3 + n_markers]).astype('int'),
    #     'mid': np.array([4, 5, 4 + n_markers, 5 + n_markers]).astype('int'),
    #     'front': np.array([6, 7, 6 + n_markers, 7 + n_markers]).astype('int')}
    idxs['legs'] = np.concatenate([idxs['back'], idxs['mid'], idxs['front']])

    # running - look at ball me
    zs_run = np.zeros_like(ball_me)
    ball_me2 = ball_me - np.percentile(ball_me, 1)
    zs_run[ball_me2 >= walk_thresh] = 1

    # still - threshold data to get moving/non-moving
    me = np.concatenate([np.zeros((1, D)), np.square(np.diff(markers, axis=0))], axis=0)

    xs_me = np.mean(me, axis=1)
    xs_me = denoise_tv_chambolle(xs_me, weight=0.05)
    xs_me -= np.min(xs_me)
    xs_me /= np.percentile(xs_me, 99)
    zs_still = np.zeros_like(xs_me)
    zs_still[(xs_me < still_thresh) & ~zs_run.astype('bool')] = 1

    not_run_or_still = ~zs_run.astype('bool') & ~zs_still.astype('bool')

    # front/back groom
    xs_fg = np.mean(me[:, idxs['front']], axis=1)
    xs_fg = denoise_tv_chambolle(xs_fg, weight=0.05)
    xs_fg -= np.min(xs_fg)
    xs_fg /= np.percentile(xs_fg, 99)

    xs_bg = np.mean(me[:, idxs['back']], axis=1)
    xs_bg = denoise_tv_chambolle(xs_bg, weight=0.05)
    xs_bg -= np.min(xs_bg)
    xs_bg /= np.percentile(xs_bg, 99)

    zs_bg = np.zeros_like(xs_bg)
    zs_bg[(xs_bg >= groom_thresh) & (xs_fg < 0.02) & not_run_or_still] = 1

    zs_fg = np.zeros_like(xs_fg)
    zs_fg[(xs_fg >= groom_thresh) & (xs_bg < 0.02) & not_run_or_still] = 1

    not_grooming = ~zs_bg.astype('bool') & ~zs_fg.astype('bool')

    # abdomen-move
    xs_ab = np.mean(me[:, idxs['body']], axis=1)
    xs_ab_d = denoise_tv_chambolle(xs_ab, weight=0.05)
    xs_ab_d -= np.min(xs_ab_d)
    xs_ab_d /= np.percentile(xs_ab_d, 99)
    zs_ab = np.zeros_like(xs_ab_d)
    zs_ab[(xs_ab_d >= ab_thresh) & not_grooming & not_run_or_still] = 1

    # fidget
    xs_legs = np.max(me[:, np.concatenate([idxs['back'], idxs['mid'], idxs['front']])], axis=1)
    xs_legs = denoise_tv_chambolle(xs_legs, weight=0.05)
    xs_legs -= np.min(xs_legs)
    xs_legs /= np.percentile(xs_legs, 99)

    zs_legs = np.zeros_like(xs_legs)
    zs_legs[
        (xs_legs >= fidget_thresh) &
        not_grooming &
        ~zs_ab.astype('bool') &
        not_run_or_still &
        (ball_me2 < 0.2)] = 1
    # only allow fidgets of >x timepoints
    beg_idx = None
    end_idx = None
    for i in range(len(zs_legs)):
        if zs_legs[i] == 1:
            if beg_idx is None:
                beg_idx = i
        else:
            if beg_idx is not None:
                end_idx = i
            if (beg_idx is not None) and (end_idx is not None) and (end_idx - beg_idx < 10):
                zs_legs[beg_idx:end_idx] = 0
            beg_idx = None
            end_idx = None

    # collect states
    states = np.zeros_like(zs_still, dtype='int')  # default state = 0: unclassified
    states[zs_still == 1] = 1
    states[zs_run == 1] = 2
    states[zs_fg == 1] = 3
    states[zs_bg == 1] = 4
    states[zs_ab == 1] = 5
    states[zs_legs == 1] = 6
    state_mapping = {
        0: 'undefined',
        1: 'still',
        2: 'walk',
        3: 'front_groom',
        4: 'back_groom',
        5: 'abdomen_move',
        6: 'fidget'}

    K = len(state_mapping)

    # smooth states with categorical HMM
    if run_smoother:
        np.random.seed(0)
        models = []
        lps = []
        for i in range(n_restarts):
            models.append(HMM(K=K, D=1, observations='categorical', observation_kwargs=dict(C=K)))
            lps.append(models[i].fit(states[:, None], num_iters=150, tolerance=1e-2))
        best_idx = np.argmax([lp[-1] for lp in lps])
        model = models[best_idx]

        states_ = model.most_likely_states(states[:, None])
        model.permute(find_permutation(states, states_))
        states_new = model.most_likely_states(states[:, None])
    else:
        states_new = np.copy(states)

    return states_new, state_mapping


def ibl(
        markers, markers_nose, wheel_vel, smooth=True, still_thresh=0.01, move_thresh=0.05,
        paw_nose_dist_thresh=100, wheel_move_thresh=0.5, n_hmms=2, bout_thresh=5):
    """Compute heuristics for 4 ibl behaviors: still, move, wheel turn, groom.

    NOTE: needs access to external data (wheel velocity, nose markers)
    TODO: Dropbox data not complete enough to recreate heuristic labels for fly data

    Parameters
    ----------
    markers : np.ndarray
    markers_nose : np.ndarray
    wheel_vel : array-like
    smooth : bool, optional
        smooth data with hmm if True and n_hmms > 0; if True and n_hmms = 0, drop bouts with
        duration less than bout_thresh
    still_thresh : float
        number in [0, 1]; "still" state is present when normalized paw motion energy is below this
        threshold
    move_thresh : float
        number in [0, 1]; "move" state is present when normalized paw motion energy is above this
        threshold (and not in grooming state, and not turning wheel)
    paw_nose_dist_thresh : float
        pixels; "groom" state is present when the distance between paw and nose markers are below
        this threshold
    wheel_move_thresh : float
        number in [0, 1]; "wheel turn" state is present when, among other criteria, the normalized
        wheel speed is above this threshold
    n_hmms : int
        number of randomly initialized hmms to fit (select best) if smooth=True
    bout_thresh : int
        minimum bout length to keep if smooth=True and n_hmms=0

    Returns
    -------
    dict

    """

    assert move_thresh >= still_thresh  # doesn't make sense otherwise

    from skimage.restoration import denoise_tv_chambolle
    from ibl_utils.markers import compute_distance

    T, D = markers.shape

    # failure modes:
    # still:
    # turning:
    # moving:
    # grooming:
    # misc:

    me = np.concatenate([np.zeros((1, D)), np.square(np.diff(markers, axis=0))], axis=0)
    dist_paw_nose = compute_distance(markers, np.nanmedian(markers_nose, axis=0))

    # still - threshold paw me to get moving/non-moving
    paw_me = np.mean(me, axis=1)
    paw_me_d = denoise_tv_chambolle(paw_me, weight=0.05)
    paw_me_d -= np.min(paw_me_d)
    paw_me_d /= np.percentile(paw_me_d, 99)
    zs_still = np.zeros_like(paw_me_d)
    zs_still[(paw_me_d < still_thresh) & (dist_paw_nose > paw_nose_dist_thresh)] = 1

    # front groom - paw marker is close to nose marker
    zs_groom = np.zeros_like(dist_paw_nose)
    zs_groom[dist_paw_nose < paw_nose_dist_thresh] = 1

    # turning - paw me and wheel velocity above thresholds, paw far away from nose
    wh_vel_01 = np.abs(wheel_vel)
    wh_vel_01 -= np.min(wh_vel_01)
    wh_vel_01 /= np.percentile(wh_vel_01, 99)
    zs_turning = np.zeros_like(wh_vel_01)
    zs_turning[
        (wh_vel_01 > wheel_move_thresh) &
        (paw_me_d > still_thresh) &
        (dist_paw_nose > paw_nose_dist_thresh)] = 1

    # moving - paw me above threshold, wheel velocity below threshold, paw far away from nose
    zs_moving = np.zeros_like(wh_vel_01)
    zs_moving[
        (wh_vel_01 < wheel_move_thresh) &
        (paw_me_d > move_thresh) &
        (dist_paw_nose > paw_nose_dist_thresh)] = 1

    # collect states
    labels = np.zeros_like(zs_still, dtype='int')  # default state = 0: undefined
    labels[zs_still == 1] = 1
    labels[zs_moving == 1] = 2
    labels[zs_turning == 1] = 3
    labels[zs_groom == 1] = 4
    label_mapping = {
        0: 'undefined',
        1: 'still',
        2: 'move',
        3: 'wheel_turn',
        4: 'groom'}

    if smooth and n_hmms > 0:
        # smooth states with categorical HMM (best of n)
        from ssm import HMM
        from ssm.util import find_permutation
        np.random.seed(0)
        models = []
        lps = []
        for i in range(n_hmms):
            models.append(HMM(
                K=len(label_mapping),
                D=1,
                observations='categorical',
                observation_kwargs=dict(C=len(label_mapping))))
            lps.append(models[i].fit(labels[:, None], num_iters=150, tolerance=1e-2))
        best_idx = np.argmax([lp[-1] for lp in lps])
        model = models[best_idx]

        labels_ = model.most_likely_states(labels[:, None])
        model.permute(find_permutation(labels, labels_))
        labels_smooth = model.most_likely_states(labels[:, None])
    elif smooth and n_hmms == 0:
        # only take runs longer than `bout_thresh` frames
        idx_list = _get_state_runs([labels])
        # loop through classes
        for idxs in idx_list:
            # loop through instances
            for idx in idxs:
                if idx[2] - idx[1] < bout_thresh:
                    labels[idx[1]:idx[2]] = 0  # set to background
        labels_smooth = labels
    else:
        labels_smooth = labels

    return labels_smooth, label_mapping


def oft(
        markers, pixels, boundary_thresh=5, rear_thresh=50, speed_thresh=1, shadow_thresh=1,
        smooth=True, bout_thresh=5):
    """Compute heuristics for 3 oft behaviors: supported rear, unsupported rear, grooming.

    NOTE: needs to be run on raw markers (which includes arena corners), not aligned markers

    Parameters
    ----------
    markers : np.ndarray
    pixels : np.ndarray
        shape (T, 2), first column is n_pixels mouse, second column is n_pixels shadow
    boundary_thresh : float, optional
    rear_thresh : float, optional
    speed_thresh : float, optional
    shadow_thresh : float, optional
    smooth : bool, optional
        if True, smooth data by dropping bouts with duration less than bout_thresh
    bout_thresh : int, optional
        minimum bout length to keep if smooth=True and n_hmms=0

    Returns
    -------
    dict

    """
    from daart_utils.utils import point_to_line_distance, point_to_point_distance

    # ------------------------------------------------------
    # supported rear
    # ------------------------------------------------------
    # - distance of mouse to boundary is below a threshold
    # - distance of bodycentre to neck is below a threshold

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

    # compute distance between bodycentre and neck
    # neck_to_body = point_to_point_distance(markers['bodycentre'], markers['neck'])
    # head_to_neck = point_to_point_distance(markers['headcentre'], markers['neck'])
    # nose_to_head = point_to_point_distance(markers['nose'], markers['headcentre'])
    # nose_to_neck = point_to_point_distance(markers['nose'], markers['neck'])
    nose_to_tail = point_to_point_distance(markers['nose'], markers['tailbase'])
    speed = np.concatenate([
        [0],
        point_to_point_distance(markers['bodycentre'][:-1], markers['bodycentre'][1:])])

    zs_supp = np.zeros((markers['nose'].shape[0],))
    zs_supp[(mouse_to_boundary < boundary_thresh) & (nose_to_tail < rear_thresh)] = 1

    # ------------------------------------------------------
    # unsupported rear
    # ------------------------------------------------------
    # - distance of mouse to boundary is above a threshold (10-20); should be large to avoid
    #   reflections being captured in connected component
    # - speed of mouse is low (<2)
    # - mouse pixels are small (<800) OR (shadow/body) ratio is > 1
    zs_unsupp = np.zeros_like(zs_supp)  # no unsupported rear labels for now
    shadow_body_ratio = pixels[:, 1] / pixels[:, 0]
    zs_unsupp[
        (mouse_to_boundary > 20)
        & (speed < speed_thresh)
        & (shadow_body_ratio > shadow_thresh)
    ] = 1

    # ------------------------------------------------------
    # grooming
    # ------------------------------------------------------
    zs_groom = np.zeros_like(zs_supp)
    height_width_ratio = pixels[:, 2] / pixels[:, 3]
    zs_groom[
        (mouse_to_boundary > 20)
        & (speed < speed_thresh / 2)
        & (height_width_ratio > 1)
        & ~zs_unsupp.astype('bool')
    ] = 1

    # ------------------------------------------------------
    # collect states
    # ------------------------------------------------------
    labels = np.zeros_like(zs_supp, dtype='int')  # default state = 0: undefined
    labels[zs_supp == 1] = 1
    labels[zs_unsupp == 1] = 2
    labels[zs_groom == 1] = 3
    label_mapping = {
        0: 'undefined',
        1: 'supported',
        2: 'unsupported',
        3: 'grooming'}

    if smooth and bout_thresh > 0:
        # only take runs longer than `bout_thresh` frames
        idx_list = get_label_runs([labels])
        # loop through classes
        for idxs in idx_list:
            # loop through instances
            for idx in idxs:
                if idx[2] - idx[1] < bout_thresh:
                    labels[idx[1]:idx[2]] = 0  # set to background
        labels_smooth = labels
    else:
        labels_smooth = labels

    return labels_smooth, label_mapping


def resident_intruder_old(
        features, feature_names, anogenital_thresh=10, int_sniff_thresh=20, smooth=True,
        bout_thresh=5):
    """Compute heuristics for resident-intruder behaviors.

    NOTE: needs to be run on simba features, not markers

    Parameters
    ----------
    features : np.ndarray
    feature_names : np.ndarray
    anogenital_thresh : float, optional
    int_sniff_thresh : float, optional
    smooth : bool, optional
        if True, smooth data by dropping bouts with duration less than bout_thresh
    bout_thresh : int, optional
        minimum bout length to keep if smooth=True and n_hmms=0

    Returns
    -------
    dict

    """

    # ------------------------------------------------------
    # anogenital_sniffing
    # ------------------------------------------------------
    idx_feature = np.where(feature_names == 'M1_Nose_to_M2_tail_base')[0][0]
    zs_agsniff = np.zeros((features.shape[0],))
    zs_agsniff[features[:, idx_feature] < anogenital_thresh] = 1

    # ------------------------------------------------------
    # intruder_sniff
    # ------------------------------------------------------
    idx_feature = np.where(feature_names == 'M2_Nose_to_M1_tail_base')[0][0]
    zs_intsniff = np.zeros_like(zs_agsniff)
    zs_intsniff[features[:, idx_feature] < int_sniff_thresh] = 1

    # ------------------------------------------------------
    # collect states
    # ------------------------------------------------------
    labels = np.zeros_like(zs_agsniff, dtype='int')  # default state = 0: undefined
    labels[zs_agsniff == 1] = 2
    labels[zs_intsniff == 1] = 6
    label_mapping = {
        0: 'undefined',
        1: 'attack',
        2: 'anogenital_sniffing',
        3: 'mounting',
        4: 'allogrooming_normal',
        5: 'allogroom_vigorous',
        6: 'intruder_sniff'}

    if smooth and bout_thresh > 0:
        # only take runs longer than `bout_thresh` frames
        idx_list = get_label_runs([labels])
        # loop through classes
        for idxs in idx_list:
            # loop through instances
            for idx in idxs:
                if idx[2] - idx[1] < bout_thresh:
                    labels[idx[1]:idx[2]] = 0  # set to background
        labels_smooth = labels
    else:
        labels_smooth = labels

    return labels_smooth, label_mapping


def resident_intruder(
        features, feature_names, centroid_thresh=120, ano_sniff_thresh=10, mount_thresh=5,
        int_sniff_thresh=10, tot_move_cent_thresh=10, tot_move_dev_thresh=-300,
        nose_move_thresh=5, smooth=True, bout_thresh=5):
    """Compute heuristics for resident-intruder behaviors.

    NOTE: needs to be run on simba features, not markers

    Parameters
    ----------
    features : np.ndarray
    feature_names : list
    []_thresh : float, optional
    smooth : bool, optional
        if True, smooth data by dropping bouts with duration less than bout_thresh
    bout_thresh : int, optional
        minimum bout length to keep if smooth=True and n_hmms=0

    Returns
    -------
    dict

    """

    # ------------------------------------------------------
    # other
    # ------------------------------------------------------
    idx_feature = np.where(feature_names == 'Centroid_distance')[0][0]
    zs_other = np.zeros((features.shape[0],))
    zs_other[features[:, idx_feature] > centroid_thresh] = 1
    zs_other_bin = zs_other.astype('int')

    # ------------------------------------------------------
    # anogenital_sniffing
    # ------------------------------------------------------
    idx_feature = np.where(feature_names == 'M1_Nose_to_M2_tail_base')[0][0]
    zs_anosniff = np.zeros_like(zs_other)
    zs_anosniff[(features[:, idx_feature] < ano_sniff_thresh)] = 1

    # ------------------------------------------------------
    # intruder_sniff
    # ------------------------------------------------------
    idx_feature = np.where(feature_names == 'M2_Nose_to_M1_tail_base')[0][0]
    zs_intsniff = np.zeros_like(zs_other)
    zs_intsniff[(features[:, idx_feature] < int_sniff_thresh)] = 1

    # ------------------------------------------------------
    # attack
    # ------------------------------------------------------
    idx_feature = np.where(feature_names == 'Total_movement_centroids')[0][0]
    zs_attack_idx0 = features[:, idx_feature] > tot_move_cent_thresh
    idx_feature = np.where(feature_names == 'Total_movement_all_bodyparts_both_mice_deviation')[0][
        0]
    zs_attack_idx1 = features[:, idx_feature] < tot_move_dev_thresh
    idx_feature = np.where(feature_names == 'Nose_movement_M1_median_15')[0][0]
    zs_attack_idx2 = features[:, idx_feature] > nose_move_thresh

    zs_attack = np.zeros_like(zs_other)
    zs_attack[(zs_attack_idx0 | zs_attack_idx1 | zs_attack_idx2)] = 1

    # ------------------------------------------------------
    # mounting
    # ------------------------------------------------------
    idx_feature = np.where(np.array(handler.features.names) == 'Mouse_2_Nose_to_centroid')[0][0]
    feat = features[:, idx_feature]
    feat = np.abs(np.concatenate([[0], np.diff(feat)]))
    zs_mount = np.zeros_like(zs_other)
    zs_mount[(feat > mount_thresh)] = 1

    # ------------------------------------------------------
    # collect states
    # ------------------------------------------------------
    labels = np.zeros_like(zs_other, dtype='int')  # default state = 0: background
    labels[zs_other == 1] = 1
    labels[zs_anosniff == 1] = 4
    #     labels[zs_mount == 1] = 5
    labels[zs_intsniff == 1] = 7
    labels[zs_attack == 1] = 8
    label_mapping = {
        0: 'background',
        1: 'other',
        2: 'face_grooming',
        3: 'body_grooming',
        4: 'anogenital_sniffing',
        5: 'mounting',
        6: 'vigorous_grooming',
        7: 'intruder_sniffing',
        8: 'attack',
    }

    if smooth and bout_thresh > 0:
        # only take runs longer than `bout_thresh` frames
        idx_list = get_label_runs([labels])
        # loop through classes
        for idxs in idx_list:
            # loop through instances
            for idx in idxs:
                if idx[2] - idx[1] < bout_thresh:
                    labels[idx[1]:idx[2]] = 0  # set to background
        labels_smooth = labels
    else:
        labels_smooth = labels

    return labels_smooth, label_mapping


def calms21(
        features, feature_names, movement_dev_thresh=-400, nose2tailbase_thresh=20,
        centroid_thresh_mount=50, centroid_thresh_other=400, smooth=True, bout_thresh=5):
    """Compute heuristics for resident-intruder behaviors from CalMS21 dataset.

    NOTE: needs to be run on simba features AND markers

    Parameters
    ----------
    features : np.ndarray
    feature_names : list
    movement_dev_thresh : float, optional
    nose2tailbase_thresh : float, optional
    centroid_thresh_mount : float, optional
    centroid_thresh_other : float, optional
    smooth : bool, optional
        if True, smooth data by dropping bouts with duration less than bout_thresh
    bout_thresh : int, optional
        minimum bout length to keep if smooth=True and n_hmms=0

    Returns
    -------
    dict

    """

    # ------------------------------------------------------
    # other
    # ------------------------------------------------------
    idx_feature = np.where(feature_names == 'Centroid_distance')[0][0]
    zs_other = np.zeros((features.shape[0],))
    zs_other[features[:, idx_feature] > centroid_thresh_other] = 1

    # ------------------------------------------------------
    # mount
    # ------------------------------------------------------
    idx_feature = np.where(feature_names == 'Centroid_distance')[0][0]
    zs_mount = np.zeros_like(zs_other)
    zs_mount[features[:, idx_feature] < centroid_thresh_mount] = 1

    # ------------------------------------------------------
    # attack
    # ------------------------------------------------------
    idx_feature = np.where(
        feature_names == 'Total_movement_all_bodyparts_both_mice_deviation')[0][0]
    zs_attack = np.zeros_like(zs_other)
    zs_attack[
        (features[:, idx_feature] < movement_dev_thresh)
        & ~zs_mount.astype('bool')
        ] = 1

    # ------------------------------------------------------
    # investigation
    # ------------------------------------------------------
    idx_feature = np.where(feature_names == 'M1_Nose_to_M2_tail_base')[0][0]
    zs_inv = np.zeros_like(zs_other)
    zs_inv[
        (features[:, idx_feature] < nose2tailbase_thresh)
        & ~zs_mount.astype('bool')
        & ~zs_attack.astype('bool')] = 1

    # ------------------------------------------------------
    # collect states
    # ------------------------------------------------------
    labels = np.zeros_like(zs_other, dtype='int')  # default state = 0: undefined
    labels[zs_attack == 1] = 1
    labels[zs_inv == 1] = 2
    labels[zs_mount == 1] = 3
    labels[zs_other == 1] = 4
    label_mapping = {
        0: 'background',
        1: 'attack',
        2: 'investigation',
        3: 'mount',
        4: 'other',
    }

    if smooth and bout_thresh > 0:
        # only take runs longer than `bout_thresh` frames
        idx_list = get_label_runs([labels])
        # loop through classes
        for idxs in idx_list:
            # loop through instances
            for idx in idxs:
                if idx[2] - idx[1] < bout_thresh:
                    labels[idx[1]:idx[2]] = 0  # set to background
        labels_smooth = labels
    else:
        labels_smooth = labels

    return labels_smooth, label_mapping
