"""Data handler class for making videos and other raw data manipulation tasks."""

import cv2
import numpy as np
import os
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
                filepath = os.path.join(self.base_path, 'videos', '%s.%s' % (expt_id, ext))
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
        extensions = ['csv', 'h5', 'npy']
        if filepath is not None:
            pass
        elif self.base_path is not None:
            for ext in extensions:
                filepath = os.path.join(
                    self.base_path, 'markers', '%s_labeled.%s' % (expt_id, ext))
                if os.path.exists(filepath):
                    break
        if filepath is None:
            raise FileNotFoundError(
                'Must supply a filepath for markers with extension in {}'.format(extensions))
        self.markers.load_markers(filepath)

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
                    self.base_path, 'labels-hand', '%s_labels.%s' % (expt_id, ext))
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
                    self.base_path, 'labels-heuristic', '%s_labels.%s' % (expt_id, ext))
                if os.path.exists(filepath):
                    break
        if filepath is None:
            raise FileNotFoundError(
                'Must supply a filepath for labels with extension in {}'.format(extensions))
        self.heuristic_labels.load_labels(filepath)

    def load_model_labels(self, filepath=None):
        """Load model labels.

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
                    self.base_path, 'labels-model', '%s_labels.%s' % (expt_id, ext))
                if os.path.exists(filepath):
                    break
        if filepath is None:
            raise FileNotFoundError(
                'Must supply a filepath for labels with extension in {}'.format(extensions))
        self.model_labels.load_labels(filepath)

    def load_all_data(
            self, load_video=True, load_markers=True, load_hand_labels=True,
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
        if load_hand_labels:
            print('loading hand label data...', end='')
            self.load_hand_labels()
            print('done')
        if load_heuristic_labels:
            print('loading heuristic label data...', end='')
            self.load_heuristic_labels()
            print('done')

    def make_labeled_video(self):
        # video (optional)
        # markers (optional)
        # most likely state (optional)
        pass

    def make_syllable_video(self):
        pass


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
        self.marker_names = None

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
            self.marker_names = marker_names
            for m, marker_name in enumerate(marker_names):
                self.vals[marker_name] = np.concatenate([xs[:, m, None], ys[:, m, None]], axis=1)
                self.likelihoods[marker_name] = ls[:, m]
        elif file_ext == 'h5':
            xs, ys, ls, marker_names = load_marker_h5(filepath)
            self.marker_names = marker_names
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


class Labels(object):
    """Simple class for loading/manipulating labels."""

    def __init__(self):

        # label names
        # type : list of strs
        self.label_names = None

        # dense representation of labels
        # type : np.ndarray
        self.vals = None

        # location of labels
        # type : str
        self.path = None

        # boolean check
        self.is_loaded = False

    def load_labels(self, filepath):
        """Load labels from csv or pickle file.

        Parameters
        ----------
        filepath : str
            absolute path of label file

        """
        file_ext = filepath.split('.')[-1]

        if file_ext == 'csv':
            labels, label_names = load_label_csv(filepath)
        elif file_ext == 'pkl' or file_ext == 'pickle':
            labels, label_names = load_label_pkl(filepath)
        else:
            raise ValueError('"%s" is an invalid file extension' % file_ext)

        self.label_names = label_names
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
    data = np.genfromtxt(filepath, delimiter=',', dtype=None, encoding=None)
    marker_names = list(data[1, 1::3])
    markers = data[3:, 1:].astype('float')  # get rid of headers, etc.
    xs = markers[:, 0::3]
    ys = markers[:, 1::3]
    ls = markers[:, 2::3]
    return xs, ys, ls, marker_names


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
    import h5py
    with h5py.File(filepath, 'r') as f:
        t = f['df_with_missing']['table'][()]
    markers = np.concatenate([t[i][1][None, :] for i in range(len(t))])
    marker_names = None
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
    labels = np.genfromtxt(filepath, delimiter=',', dtype=np.int, encoding=None, skip_header=1)
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
    label_dict = data['state_mapping']
    label_names = [label_dict[i] for i in range(len(label_dict))]
    return labels, label_names
