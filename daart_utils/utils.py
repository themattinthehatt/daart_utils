"""Various preprocessing and utility functions."""

import numpy as np


# --------------------
# filters
# --------------------

def smooth_interpolate_signal_sg(signal, window=31, order=3, interp_kind='linear'):
    """Run savitzy-golay filter on signal, interpolate through nan points.

    Parameters
    ----------
    signal : np.ndarray
        original noisy signal of shape (t,), may contain nans
    window : int
        window of polynomial fit for savitzy-golay filter
    order : int
        order of polynomial for savitzy-golay filter
    interp_kind : str
        type of interpolation for nans, e.g. 'linear', 'quadratic', 'cubic'

    Returns
    -------
    np.ndarray
        smoothed, interpolated signal for each time point, shape (t,)

    """

    from scipy.interpolate import interp1d

    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
    if len(good_idxs) < window:
        print('not enough non-nan indices to filter; returning original signal')
        return signal_noisy_w_nans
    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')

    signal = interpolater(timestamps)

    return signal


def smooth_interpolate_signal_tv(signal, tv_weight=10, interp_kind='linear'):
    """Perform total variation denoising on a 1D signal

    Parameters
    ----------
    signal : array-like
    tv_weight : int
        total variation denoising weight (higher leads to smoother outputs)
    interp_kind : str
        how to interpolate NaN points

    Returns
    -------
    np.array

    """
    from scipy.interpolate import interp1d
    from skimage.restoration import denoise_tv_chambolle

    signal_smooth_no_nans = np.nan * np.zeros(signal.shape)

    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    for i in range(signal_noisy_w_nans.shape[1]):
        good_idxs = np.where(~np.isnan(signal_noisy_w_nans[:, i]))
        # interpolate nan points
        interpolater = interp1d(
            timestamps[good_idxs], signal_noisy_w_nans[good_idxs, i],
            kind=interp_kind, fill_value='extrapolate')
        signal_noisy_no_nans = interpolater(timestamps)
        signal_smooth_no_nans[:, i] = denoise_tv_chambolle(signal_noisy_no_nans, weight=tv_weight)

    return signal_smooth_no_nans


def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    https://dsp.stackexchange.com/a/64313

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """

    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


# --------------------
# labels
# --------------------

def get_label_runs(labels):
    """Find occurrences of each discrete label.

    Adapted from:
    https://github.com/themattinthehatt/behavenet/blob/master/behavenet/plotting/arhmm_utils.py#L24

    Parameters
    ----------
    labels : list
        each entry is numpy array containing discrete label for each frame

    Returns
    -------
    list
        list of length discrete labels, each list contains all occurences of that discrete label by
        [chunk number, starting index, ending index]

    """

    max_label = np.max([np.max(s) for s in labels])
    indexing_list = [[] for _ in range(max_label + 1)]

    for i_chunk, chunk in enumerate(labels):

        # pad either side so we get start and end chunks
        chunk = np.pad(chunk, (1, 1), mode='constant', constant_values=-1)
        # don't add 1 because of start padding, now index in original unpadded data
        split_indices = np.where(np.ediff1d(chunk) != 0)[0]
        # last index will be 1 higher that it should be due to padding
        split_indices[-1] -= 1

        for i in range(len(split_indices) - 1):

            # get which label this chunk was (+1 because data is still padded)
            which_label = chunk[split_indices[i] + 1]

            indexing_list[which_label].append([i_chunk, split_indices[i], split_indices[i + 1]])

    # convert lists to numpy arrays
    indexing_list = [np.asarray(indexing_list[i_label]) for i_label in range(max_label + 1)]

    return indexing_list


# --------------------
# geometry
# --------------------

def point_to_point_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2, axis=1))


def point_to_line_distance(pt, line_pt_1, line_pt_2):
    return np.cross(line_pt_2 - line_pt_1, pt - line_pt_1) / np.linalg.norm(line_pt_2 - line_pt_1)


def polygon_area(pt1, pt2, pt3, pt4=None):
    """Area of the polygon described by the given 2D points."""
    assert pt1.shape[1] == 2
    assert pt2.shape[1] == 2
    assert pt3.shape[1] == 2
    if pt4 is None:
        area = 0.5 * np.abs(
            pt1[:, 0] * pt2[:, 1] - pt1[:, 1] * pt2[:, 0] +
            pt2[:, 0] * pt3[:, 1] - pt2[:, 1] * pt3[:, 0] +
            pt3[:, 0] * pt1[:, 1] - pt3[:, 1] * pt1[:, 0])
    else:
        area = 0.5 * np.abs(
            pt1[:, 0] * pt2[:, 1] - pt1[:, 1] * pt2[:, 0] +
            pt2[:, 0] * pt3[:, 1] - pt2[:, 1] * pt3[:, 0] +
            pt3[:, 0] * pt4[:, 1] - pt3[:, 1] * pt4[:, 0] +
            pt4[:, 0] * pt1[:, 1] - pt4[:, 1] * pt1[:, 0])
    return area


def absolute_angle(pt1, pt2, pt3):
    """Angle between the 2D vectors pt1-pt2 and pt3-pt2."""
    assert pt1.shape[1] == 2
    assert pt2.shape[1] == 2
    assert pt3.shape[1] == 2
    a = pt1 - pt2
    b = pt3 - pt2
    det = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
    dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    angle = np.arctan(det / dot)
    return angle
