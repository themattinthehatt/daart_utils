"""Data handler class for making videos and other raw data manipulation tasks."""

import cv2
import numpy as np
import os
import pandas as pd
import pickle


class DataHandler(object):
    """Class for manipulating video, markers, and discrete labels.

    Notes

    """

    def __init__(self, session_id, base_path=None):
        """Initialize data handler.

        Parameters
        ----------
        session_id : str
            name of session
        base_path : str, optional
            assumes video, markers, and discrete labels are stored in the following way:
            - videos: <base_path>/videos/<session_id>.[avi/mp4]
            - markers: <base_path>/markers/<session_id>_labeled.[h5/csv]
            - hand labels: <base_path>/labels-hand/<session_id>_labeled.csv
            - heuristic labels: <base_path>/labels-heuristic/<session_id>_labeled.[csv/pkl]

        """

        # --------------------------------
        # metadata
        # --------------------------------
        self.session_id = session_id
        self.base_path = base_path

        # --------------------------------
        # initialize data objects
        # --------------------------------
        # object to handle video data
        self.video = Video()

        # object to handle 2D marker data
        self.markers = Markers()

        # object to handle feature data
        self.features = Features()

        # object to handle hand labels
        self.hand_labels = Labels()

        # object to handle heuristic labels
        self.heuristic_labels = Labels()

        # object to handle model labels
        self.model_labels = Labels()

    def load_video(self, filepath=None):
        """Load video data.

        Parameters
        ----------
        filepath : str, optional
            use this to override automatic path computation

        """
        extensions = ['mp4', 'avi']
        if filepath is not None:
            pass
        elif self.base_path is not None:
            for ext in extensions:
                filepath = os.path.join(self.base_path, 'videos', '%s.%s' % (self.session_id, ext))
                if os.path.exists(filepath):
                    break
        if filepath is None:
            raise FileNotFoundError(
                'Must supply a filepath for video with extension in {}'.format(extensions))
        self.video.load_video_cap(filepath)

    def load_markers(self, filepath=None):
        """Load markers.

        Parameters
        ----------
        filepath : str, optional
            use this to override automatic path computation

        """

        if filepath is not None:
            pass
        elif self.base_path is not None:
            filepath = self.get_marker_filepath()
        else:
            raise FileNotFoundError('Must supply a marker filepath if base_path not defined')
        self.markers.load_markers(filepath)

    def load_features(self, filepath=None, dirname='features'):
        """Load features (often derived from markers).

        Parameters
        ----------
        filepath : str, optional
            use this to override automatic path computation
        dirname : str, optional
            features located in `self.path_base/dirname/feature_file.csv`

        """

        if filepath is not None:
            pass
        elif self.base_path is not None:
            filepath = self.get_feature_filepath(dirname=dirname)
        else:
            raise FileNotFoundError('Must supply a feature filepath if base_path not defined')
        self.features.load_features(filepath)

    def load_hand_labels(self, filepath=None):
        """Load hand labels.

        Parameters
        ----------
        filepath : str, optional
            use this to override automatic path computation

        """
        extensions = ['csv']
        if filepath is not None:
            pass
        elif self.base_path is not None:
            for ext in extensions:
                filepath = os.path.join(
                    self.base_path, 'labels-hand', '%s_labels.%s' % (self.session_id, ext))
                if os.path.exists(filepath):
                    break
        if filepath is None:
            raise FileNotFoundError(
                'Must supply a filepath for labels with extension in {}'.format(extensions))
        self.hand_labels.load_labels(filepath)

    def load_heuristic_labels(self, filepath=None):
        """Load hand labels.

        Parameters
        ----------
        filepath : str, optional
            use this to override automatic path computation

        """
        extensions = ['csv', 'pkl', 'pickle']
        if filepath is not None:
            pass
        elif self.base_path is not None:
            for ext in extensions:
                filepath = os.path.join(
                    self.base_path, 'labels-heuristic', '%s_labels.%s' % (self.session_id, ext))
                if os.path.exists(filepath):
                    break
        if filepath is None:
            raise FileNotFoundError(
                'Must supply a filepath for labels with extension in {}'.format(extensions))
        self.heuristic_labels.load_labels(filepath)

    def load_model_labels(self, filepath=None, logits=True):
        """Load model labels.

        Parameters
        ----------
        filepath : str, optional
            use this to override automatic path computation
        logits : bool
            True if loaded values are raw logits; will be processed into one-hot vector

        """
        extensions = ['csv', 'pkl', 'pickle']
        if filepath is not None:
            pass
        elif self.base_path is not None:
            for ext in extensions:
                filepath = os.path.join(
                    self.base_path, 'labels-model', '%s_labels.%s' % (self.session_id, ext))
                if os.path.exists(filepath):
                    break
        if filepath is None:
            raise FileNotFoundError(
                'Must supply a filepath for labels with extension in {}'.format(extensions))
        self.model_labels.load_labels(filepath, logits=logits)

    def load_all_data(
            self, load_video=True, load_markers=True, load_features=False, load_hand_labels=True,
            load_heuristic_labels=True):
        """Helper function to load video, markers, hand and heuristic labels."""
        if load_video:
            print('loading video data...', end='')
            self.load_video()
            print('done')
        if load_markers:
            print('loading marker data...', end='')
            self.load_markers()
            print('done')
        if load_features:
            print('loading features data...', end='')
            self.load_features()
            print('done')
        if load_hand_labels:
            print('loading hand label data...', end='')
            self.load_hand_labels()
            print('done')
        if load_heuristic_labels:
            print('loading heuristic label data...', end='')
            self.load_heuristic_labels()
            print('done')

    def get_marker_filepath(self):
        """Search over different file extensions for markers."""
        filepath = None
        extensions = ['csv', 'h5', 'npy']
        for ext in extensions:
            filepath = os.path.join(
                self.base_path, 'markers', '%s_labeled.%s' % (self.session_id, ext))
            if os.path.exists(filepath):
                break
        if filepath is None:
            raise FileNotFoundError(
                'Must supply a filepath for markers with extension in {}'.format(extensions))
        return filepath

    def get_feature_filepath(self, dirname='features'):
        """Search over different file extensions for features."""
        filepath = None
        extensions = ['csv']
        for ext in extensions:
            filepath = os.path.join(
                self.base_path, dirname, '%s_labeled.%s' % (self.session_id, ext))
            if os.path.exists(filepath):
                break
        if filepath is None:
            raise FileNotFoundError(
                'Must supply a filepath for features with extension in {}'.format(extensions))
        return filepath

    def make_labeled_video(
            self, save_file, idxs, include_markers=True, label_type='none', framerate=20,
            height=4):
        """Export raw video overlaid with markers.

        Parameters
        ----------
        save_file : str
            absolute filename of path (including extension)
        idxs : array-like
            array of indices for video
        include_markers : bool, optional
            True to overlay markers on video
        label_type : str, optional
            select label that will appear in corner of video
            'none' | 'hand' | 'heuristic' | 'model'
        framerate : float, optional
            framerate of video
        height : float, optional
            height of movie in inches

        """

        from daart_utils.videos import make_labeled_video

        # select data for plotting
        frames = self.video.get_frames_from_idxs(idxs)

        if include_markers:
            if not self.markers.is_loaded:
                raise ValueError('Cannot include markers if they are not loaded')
            markers = {m: self.markers.vals[m][idxs] for m in self.markers.names}
        else:
            markers = None

        if label_type == 'hand':
            labels = {name: self.hand_labels.vals[idxs, l]
                      for l, name in enumerate(self.hand_labels.names)}
        elif label_type == 'heuristic':
            labels = {name: self.heuristic_labels.vals[idxs, l]
                      for l, name in enumerate(self.heuristic_labels.names)}
        elif label_type == 'model':
            labels = {name: self.model_labels.vals[idxs, l]
                      for l, name in enumerate(self.model_labels.names)}
        elif label_type == 'none' or label_type is None:
            labels = None
        else:
            raise NotImplementedError('must choose from "none", "hand", "heuristic", "model"')

        make_labeled_video(
            save_file=save_file, frames=frames, markers=markers, labels=labels,
            framerate=framerate, height=height)

    def make_syllable_video(
            self, save_file, label_type, include_markers=False, smooth_markers=False, markersize=8,
            save_states_separately=False, min_threshold=5, n_buffer=5, n_pre_frames=3,
            max_frames=1000, framerate=20):
        """

        Parameters
        ----------
        save_file : str
            absolute path of filename (including extension)
        label_type : str
            label type from which to extract syllables
            'hand' | 'heuristic' | 'model'
        include_markers : bool or list
            True to overlay markers on video, or a list of marker names to include
        smooth_markers : bool
            True to first smooth markers with a savitzky-golay filter
        markersize : float, optional
            size of markers if plotted
        save_states_separately : bool
            True to make a video for each state; False to combine into a multi-panel video
        min_threshold : int, optional
            minimum length of label clips
        n_buffer : int, optional
            number of black frames between clips
        n_pre_frames : int, optional
            nunber of frames before syllable onset
        max_frames : int, optional
            length of video
        framerate : float, optional
            framerate of video

        """

        from daart_utils.utils import smooth_interpolate_signal_sg
        from daart_utils.videos import make_syllable_video

        if label_type == 'hand':
            labels = self.hand_labels.vals
            names = self.hand_labels.names
        elif label_type == 'heuristic':
            labels = self.heuristic_labels.vals
            names = self.heuristic_labels.names
        elif label_type == 'model':
            print('warning! probably want to threshold probabilities; need to implement')
            labels = self.model_labels.vals
            names = self.model_labels.names
        else:
            raise NotImplementedError('must choose from "hand", "heuristic", "model"')

        # assume a single behavior per frame (can generalize later)
        labels = np.argmax(labels, axis=1)
        label_mapping = {l: name for l, name in enumerate(names)}

        if include_markers:
            if not self.markers.is_loaded:
                raise ValueError('Cannot include markers if they are not loaded')
            if isinstance(include_markers, list):
                markers = {m: np.copy(self.markers.vals[m]) for m in include_markers}
            else:
                markers = {m: np.copy(self.markers.vals[m]) for m in self.markers.names}
            for name, val in markers.items():
                markers[name][self.markers.likelihoods[name] < 0.75, :] = np.nan
                if smooth_markers:
                    for i in [0, 1]:
                        markers[name][:, i] = smooth_interpolate_signal_sg(
                            markers[name][:, i], window=31, order=3,
                            interp_kind='linear')
        else:
            markers = None

        if save_states_separately:
            for n in range(np.max(labels)):
                make_syllable_video(
                    save_file=save_file, labels=labels, video_obj=self.video, markers=markers,
                    markersize=markersize, min_threshold=min_threshold, n_buffer=n_buffer,
                    n_pre_frames=n_pre_frames, max_frames=max_frames, single_label=n,
                    label_mapping=label_mapping, framerate=framerate)
        else:
            make_syllable_video(
                save_file=save_file, labels=labels, video_obj=self.video, markers=markers,
                markersize=markersize, min_threshold=min_threshold, n_buffer=n_buffer,
                n_pre_frames=n_pre_frames, max_frames=max_frames, single_label=None,
                label_mapping=label_mapping, framerate=framerate)


class Video(object):
    """Simple class for loading/manipulating videos."""

    def __init__(self):

        # opencv video capture
        # type : cv2.VideoCapture object
        self.cap = None

        # location of opencv video capture
        # type : str
        self.path = None

        # total frames
        # type : int
        self.n_frames = None

        # frame width (pixels)
        # type : int
        self.frame_width = None

        # frame height (pixels)
        # type : int
        self.frame_height = None

        # boolean check
        self.is_loaded = False

    def load_video_cap(self, filepath):
        """Initialize opencv video capture objects from video file.

        Parameters
        ----------
        filepath : str
            absolute location of video (.mp4, .avi)

        """

        # load video cap
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise IOError('error opening video file at %s' % filepath)

        # save frame info
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # save filepath
        self.path = filepath

        self.is_loaded = True

    def get_frames_from_idxs(self, idxs):
        """Helper function to load video segments.

        Parameters
        ----------
        idxs : array-like
            frame indices into video

        Returns
        -------
        np.ndarray
            returned frames of shape shape (n_frames, n_channels, ypix, xpix)

        """
        is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
        n_frames = len(idxs)
        for fr, i in enumerate(idxs):
            if fr == 0 or not is_contiguous:
                self.cap.set(1, i)
            ret, frame = self.cap.read()
            if ret:
                if fr == 0:
                    height, width, _ = frame.shape
                    frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
                frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print(
                    'warning! reached end of video; returning blank frames for remainder of ' +
                    'requested indices')
                break
        return frames


class Markers(object):
    """Simple class for loading/manipulating markers."""

    def __init__(self):

        # marker names
        # type : list of strs
        self.names = None

        # 2D markers
        # type : dict with keys `self.marker_names`, vals are arrays of shape (n_t, 2)
        self.vals = {}

        # 2D marker likelihoods
        # type : dict with keys `self.marker_names`, vals are arrays of shape (n_t,)
        self.likelihoods = {}

        # location of markers
        # type : str
        self.path = None

        # boolean check
        self.is_loaded = False

    def load_markers(self, filepath):
        """Load markers from csv or h5 file.

        Parameters
        ----------
        filepath : str
            absolute path of marker file

        """
        file_ext = filepath.split('.')[-1]

        if file_ext == 'csv':
            xs, ys, ls, marker_names = load_marker_csv(filepath)
            self.names = marker_names
            for m, marker_name in enumerate(marker_names):
                self.vals[marker_name] = np.concatenate([xs[:, m, None], ys[:, m, None]], axis=1)
                self.likelihoods[marker_name] = ls[:, m]
        elif file_ext == 'h5':
            xs, ys, ls, marker_names = load_marker_h5(filepath)
            self.names = marker_names
            for m, marker_name in enumerate(marker_names):
                self.vals[marker_name] = np.concatenate([xs[:, m, None], ys[:, m, None]], axis=1)
                self.likelihoods[marker_name] = ls[:, m]
        elif file_ext == 'npy':
            raise NotImplementedError
            # assume single array
            # vals = np.load(markers_path)
            # marker_names = None
        else:
            raise ValueError('"%s" is an invalid file extension' % file_ext)

        # save filepath
        self.path = filepath

        self.is_loaded = True

    def smooth_interpolate_sg(
            self, likelihood_thresh=0.75, window=31, order=3, interp_kind='linear'):
        """Run savitzy-golay filter on markers, and interpolate through low-likelihood points.

        Parameters
        ----------
        likelihood_thresh : float, optional
            interpolate through timepoints where likelihoods are below threshold
        window : int
            window of polynomial fit for savitzy-golay filter
        order : int
            order of polynomial for savitzy-golay filter
        interp_kind : str
            type of interpolation for nans, e.g. 'linear', 'quadratic', 'cubic'

        Returns
        -------
        dict

        """
        from daart_utils.utils import smooth_interpolate_signal_sg
        markers_tmp = {m: np.copy(self.vals[m]) for m in self.names}
        # get rid of low-likelihood markers
        for name, val in markers_tmp.items():
            markers_tmp[name][self.likelihoods[name] < likelihood_thresh, :] = np.nan
            for i in [0, 1]:
                markers_tmp[name][:, i] = smooth_interpolate_signal_sg(
                    markers_tmp[name][:, i], window=window, order=order, interp_kind=interp_kind)
        return markers_tmp


class Features(object):
    """Simple class for loading/manipulating behavioral features."""

    def __init__(self):

        # marker names
        # type : list of strs
        self.names = None

        # features
        # type : np.ndarray
        self.vals = None

        # location of features
        # type : str
        self.path = None

        # boolean check
        self.is_loaded = False

    def load_features(self, filepath):
        """Load features from csv file.

        Parameters
        ----------
        filepath : str
            absolute path of feature file

        """
        file_ext = filepath.split('.')[-1]

        if file_ext == 'csv':
            vals, names = load_feature_csv(filepath)
            self.vals = vals
            self.names = names
        else:
            raise ValueError('"%s" is an invalid file extension' % file_ext)

        # save filepath
        self.path = filepath

        self.is_loaded = True


class Labels(object):
    """Simple class for loading/manipulating labels."""

    def __init__(self):

        # label names
        # type : list of strs
        self.names = None

        # dense representation of labels
        # type : np.ndarray
        self.vals = None

        # location of labels
        # type : str
        self.path = None

        # boolean check
        self.is_loaded = False

    def load_labels(self, filepath, logits=False):
        """Load labels from csv or pickle file.

        Parameters
        ----------
        filepath : str
            absolute path of label file
        logits : bool
            True if loaded values are raw logits; will be processed into one-hot vector

        """
        file_ext = filepath.split('.')[-1]

        if file_ext == 'csv':
            labels, label_names = load_label_csv(filepath)
        elif file_ext == 'pkl' or file_ext == 'pickle':
            labels, label_names = load_label_pkl(filepath)
        else:
            raise ValueError('"%s" is an invalid file extension' % file_ext)

        if logits:
            # we need to convert from logits one-hot vector
            from daart.transforms import MakeOneHot
            most_likely = np.argmax(labels, axis=1)
            labels = MakeOneHot()(most_likely)

        self.names = label_names
        self.vals = labels

        # save filepath
        self.path = filepath

        self.is_loaded = True


def load_marker_csv(filepath):
    """Load markers from csv file assuming DLC format

    --------------------------------------------------------------------------------
       scorer  | <scorer_name> | <scorer_name> | <scorer_name> | <scorer_name> | ...
     bodyparts |  <part_name>  |  <part_name>  |  <part_name>  |  <part_name>  | ...
       coords  |       x       |       y       |  likelihood   |       x       | ...
    --------------------------------------------------------------------------------
         0     |     34.5      |     125.4     |     0.921     |      98.4     | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    # data = np.genfromtxt(filepath, delimiter=',', dtype=None, encoding=None)
    # marker_names = list(data[1, 1::3])
    # markers = data[3:, 1:].astype('float')  # get rid of headers, etc.

    # define first three rows as headers (as per DLC standard)
    # drop first column ('scorer' at level 0) which just contains frame indices
    df = pd.read_csv(filepath, header=[0, 1, 2]).drop(['scorer'], axis=1, level=0)
    # collect marker names from multiindex header
    marker_names = [c[1] for c in df.columns[::3]]
    markers = df.values
    xs = markers[:, 0::3]
    ys = markers[:, 1::3]
    ls = markers[:, 2::3]
    return xs, ys, ls, marker_names


def load_feature_csv(filepath):
    """Load markers from csv file assuming the following format

    --------------------------------------------------------------------------------
        name   |     <f1>      |     <f2>      |     <f3>      |     <f4>      | ...
    --------------------------------------------------------------------------------
         0     |     34.5      |     125.4     |     0.921     |      98.4     | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    # drop first column which just contains frame indices
    df = pd.read_csv(filepath).drop(['Unnamed: 0'], axis=1)
    vals = df.values
    feature_names = list(df.columns)
    return vals, feature_names


def load_marker_h5(filepath):
    """Load markers from hdf5 file assuming DLC format

    Parameters
    ----------
    filepath : str
        absolute path of hdf5 file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    # import h5py
    # with h5py.File(filepath, 'r') as f:
    #     t = f['df_with_missing']['table'][()]
    # markers = np.concatenate([t[i][1][None, :] for i in range(len(t))])
    # marker_names = None
    # xs = markers[:, 0::3]
    # ys = markers[:, 1::3]
    # ls = markers[:, 2::3]
    import pandas as pd
    df = pd.read_hdf(filepath)
    marker_names = [d[1] for d in df.columns][0::3]
    markers = df.to_numpy()
    xs = markers[:, 0::3]
    ys = markers[:, 1::3]
    ls = markers[:, 2::3]
    return xs, ys, ls, marker_names


def load_label_csv(filepath):
    """Load labels from csv file assuming a standard format.

    --------------------------------
       | <class 0> | <class 1> | ...
    --------------------------------
     0 |     0     |     1     | ...
     1 |     0     |     1     | ...
     . |     .     |     .     | ...
     . |     .     |     .     | ...
     . |     .     |     .     | ...

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - labels (np.ndarray): shape (n_t, n_labels)
        - label names (list): name for each column in `labels` matrix

    """
    labels = np.genfromtxt(
        filepath, delimiter=',', dtype=np.int, encoding=None, skip_header=1)[:, 1:]
    label_names = list(
        np.genfromtxt(filepath, delimiter=',', dtype=None, encoding=None, max_rows=1)[1:])
    return labels, label_names


def load_label_pkl(filepath):
    """Load labels from pkl file assuming a standard format.

    Parameters
    ----------
    filepath : str
        absolute path of pickle file

    Returns
    -------
    tuple
        - labels (np.ndarray): shape (n_t, n_labels)
        - label names (list): name for each column in `labels` matrix

    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    labels = data['states']
    try:
        label_dict = data['state_mapping']
    except KeyError:
        label_dict = data['state_labels']
    label_names = [label_dict[i] for i in range(len(label_dict))]
    return labels, label_names
