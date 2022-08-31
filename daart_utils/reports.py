import numpy as np
import os
import pandas as pd
from scipy.signal import medfilt
import torch
from typing import Optional
import yaml

from daart.data import load_feature_csv, DataGenerator
from daart.eval import get_precision_recall, run_lengths, plot_training_curves
from daart.models import Segmenter
from daart.transforms import ZScore

from daart_utils.data import DataHandler, Video
from daart_utils.plotting import (
    plot_bout_histograms,
    plot_behavior_distribution,
    plot_bout_onsets_w_features,
    plot_rate_scatters,
)
from daart_utils.videos import make_labeled_video, make_syllable_video
from daart_utils.utils import get_label_runs


def update_kwargs_dict_with_defaults(kwargs_new, kwargs_default):
    for key, val in kwargs_default.items():
        if key not in kwargs_new:
            kwargs_new[key] = val
    return kwargs_new


class ReportGeneratorBase:

    def __init__(self, model_dir: str):
        """

        Parameters
        ----------
        model_dir
            directory that contains `best_val_model.pt` and `hparams.yaml` files

        """

        # save inupts
        self.model_dir = model_dir

        # load model
        self.model_file = None
        self.hparams_file = None
        self.hparams = None
        self.model = None
        self.state_names = None
        self.load_model()

    def load_model(self):

        model_file = os.path.join(self.model_dir, 'best_val_model.pt')

        hparams_file = os.path.join(self.model_dir, 'hparams.yaml')
        hparams = yaml.safe_load(open(hparams_file, 'rb'))
        hparams['device'] = 'cpu'

        model = Segmenter(hparams)
        model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        model.to(hparams['device'])
        model.eval()

        self.model_file = model_file
        self.hparams_file = hparams_file
        self.hparams = hparams
        self.model = model
        self.state_names = hparams['class_names']

    def compute_states(self, sess_id, features_file, overwrite=False):

        # define data generator signals
        signals = ['markers']
        transforms = [ZScore()]
        paths = [features_file]

        # build data generator
        hparams = self.hparams
        data_gen_test = DataGenerator(
            [sess_id], [signals], [transforms], [paths], device=hparams['device'],
            sequence_length=hparams['sequence_length'], batch_size=hparams['batch_size'],
            trial_splits=hparams['trial_splits'],
            sequence_pad=hparams['sequence_pad'], input_type=hparams['input_type'])

        # load/predict probabilities from model
        probs_file = os.path.join(self.model_dir, '%s_states.npy' % sess_id)
        if os.path.exists(probs_file) and not overwrite:
            print('loading states from %s...' % probs_file, end='')
            probs = np.load(probs_file)
            print('done')
        else:
            print('computing states for %s...' % sess_id, end='')
            tmp = self.model.predict_labels(data_gen_test, return_scores=True)
            probs = np.vstack(tmp['labels'][0])
            print('done')
            if not os.path.exists(os.path.dirname(probs_file)):
                os.makedirs(os.path.dirname(probs_file))
            np.save(probs_file, probs)

        states = np.argmax(probs, axis=1)
        states = medfilt(states, kernel_size=7).astype('int')

        return probs, states

    def generate_report(self, *args, **kwargs):
        raise NotImplementedError


class ReportGenerator(ReportGeneratorBase):
    """Generate a diagnostic report for unlabeled sessions that includes plots and videos.

    Examples
    --------
    # initialize object for unlabeled sessions
    reporter = GenerateReport(model_dir="/path/to/model_directory")
    # create report
    report_dir = reporter.generate_report(
        save_dir="/path/to/report_dir",
        features_dir="/path/to/features_directory",
        videos_dir="/path/to/videos_directory",
        format="pdf")

    Notes
    -----
    * csv files in features_directory must be named "<sess_id>_labeled.csv"
    * mp4 files in videos_directory must be names "<sess_id>.mp4"

    """

    def __init__(self, model_dir: str):
        """

        Parameters
        ----------
        model_dir
            directory that contains `best_val_model.pt` and `hparams.yaml` files

        """
        # store `model_dir` and load model
        super().__init__(model_dir=model_dir)

    @staticmethod
    def find_session_ids(features_dir):
        features_files = os.listdir(features_dir)
        sess_ids = [os.path.basename(f).split('_labeled')[0] for f in features_files]
        return sess_ids

    def generate_report(
            self,
            save_dir: str,
            features_dir: str,
            format: str = 'pdf',
            bout_example_kwargs: dict = {},
            video_kwargs: dict = {},
            video_framerate: Optional[int] = None,
            videos_dir: Optional[dir] = None,
            **kwargs
    ) -> str:
        """

        Parameters
        ----------
        save_dir
            report will be saved in `save_dir/daart-report_<sess_id>`
        features_dir
            all feature csvs in this directory will be processed
        format
            "pdf" | "png"
        bout_example_kwargs
            features_to_plot: list
            frame_win: int
            max_n_ex: int
            min_bout_len: int
        video_kwargs
            n_frames: int
            framerate: float
            markersize: float
            min_threshold: int
            n_buffer: int
            n_pre_frames: int
        video_framerate
            framerate of original video
        videos_dir
            ReportGenerator will look for videos in this directory that correspond to each feature
            csv

        Returns
        -------
        save directory

        """

        sess_ids = self.find_session_ids(features_dir)

        for sess_id in sess_ids:

            # get paths
            save_dir_full = os.path.join(save_dir, sess_id)
            os.makedirs(save_dir_full, exist_ok=True)
            if videos_dir is not None:
                video_file = os.path.join(videos_dir, '%s.mp4' % sess_id)
                videos_list = [os.path.join(videos_dir, v) for v in os.listdir(videos_dir)]
                if video_file not in videos_list:
                    video_file = None
            else:
                video_file = None

            # load features
            features_file = os.path.join(features_dir, '%s_labeled.csv' % sess_id)
            features, feature_names = load_feature_csv(features_file)

            # compute states from features
            probs, states = self.compute_states(sess_id, features_file)

            # generate diagnostic plots and videos
            generate_report(
                sess_id=sess_id,
                probs=probs,
                states=states,
                state_names=self.state_names,
                features=features,
                feature_names=feature_names,
                save_dir=save_dir_full,
                format=format,
                bout_example_kwargs=bout_example_kwargs,
                video_kwargs=video_kwargs,
                video_framerate=video_framerate,
                video_file=video_file,
            )

        return save_dir


def generate_report(
        sess_id,
        probs,
        states,
        state_names,
        features,
        feature_names,
        save_dir,
        format,
        bout_example_kwargs,
        video_kwargs,
        video_framerate,
        video_file,
):
    """Produce diagnostic plots and videos for segmenation model on unlabeled data.

    Outputs:
    * histograms of bout durations for each behavior
    * distribution of behaviors + empirical transition matrix
    * bout examples for each behavior (features, probabilities, states)
    * [optional] labeled video using part of session with lots of state switches
    * [optional] syllable video

    Parameters
    ----------
    sess_id
    probs
    states
    state_names
    features
    feature_names
    save_dir
    format
    bout_example_kwargs
    video_kwargs
    video_framerate
    video_file

    """

    # parse inputs
    bout_example_kwargs_default = {
        'features_to_plot': None,
        'frame_win': 200,
        'max_n_ex': 10,
        'min_bout_len': 5,
    }
    bout_example_kwargs = update_kwargs_dict_with_defaults(
        bout_example_kwargs, bout_example_kwargs_default)

    video_kwargs_default = {
        'max_frames': 500,
        'framerate': 20,
        'markersize': 8,
        'min_threshold': 5,
        'n_buffer': 5,
        'n_pre_frames': 3,
    }
    video_kwargs = update_kwargs_dict_with_defaults(video_kwargs, video_kwargs_default)

    # ------------------------------------------------
    # plots
    # ------------------------------------------------

    # plot histogram of bout durations
    n_cols = 3
    bouts = run_lengths(states)
    save_file = os.path.join(save_dir, 'behavior_bouts.%s' % format)
    title = 'Session %s' % sess_id
    plot_bout_histograms(
        bouts, state_names=state_names, framerate=video_framerate, n_cols=n_cols, title=title,
        save_file=save_file)

    # plot behavior distribution
    save_file = os.path.join(save_dir, 'behavior_distribution.%s' % format)
    title = 'Session %s' % sess_id
    plot_behavior_distribution(
        states, state_names=state_names, framerate=video_framerate, title=title,
        save_file=save_file)

    # plot examples of each behavior type (probs + markers + states)
    if bout_example_kwargs['features_to_plot'] is None:
        bout_example_kwargs['features_to_plot'] = feature_names
    idxs_features = np.where(
        np.isin(np.array(feature_names), np.array(bout_example_kwargs['features_to_plot'])))[0]
    features_ = features[:, idxs_features]
    bouts_w_idxs = get_label_runs([states])
    for b, bouts_list in enumerate(bouts_w_idxs):
        save_file = os.path.join(save_dir, 'ethograms_%s_onset.%s' % (state_names[b], format))
        title = '%s (%s onsets)' % (sess_id, state_names[b])
        plot_bout_onsets_w_features(
            bouts_list, markers=features_, marker_names=bout_example_kwargs['features_to_plot'],
            probs=probs, states=states, state_names=state_names, framerate=video_framerate,
            title=title, save_file=save_file, **bout_example_kwargs)

    # ------------------------------------------------
    # videos
    # ------------------------------------------------
    if video_file is not None:

        video_obj = Video()
        video_obj.load_video_cap(video_file)

        # make labeled video
        n_frames = video_kwargs['max_frames']
        # compute chunk of maximal state switching
        v = np.abs(np.diff(states))
        v[v > 0] = 1
        n_chunks = v.shape[0] // n_frames
        v_rs = np.reshape(v[:n_chunks * n_frames], (n_chunks, n_frames))
        n_changes = np.sum(v_rs, axis=1)
        best_idx = np.argmax(n_changes)
        idxs = np.arange(best_idx * n_frames, (best_idx + 1) * n_frames)
        # get frames
        frames = video_obj.get_frames_from_idxs(idxs)
        save_file = os.path.join(save_dir, 'labeled_video.mp4')
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        make_labeled_video(
            save_file, frames, frame_idxs=idxs, markers=None, probs=probs,
            state_names=state_names, height=4, **video_kwargs)

        # make syllable video
        label_mapping = {}
        for c, name in enumerate(state_names):
            label_mapping[c] = name
        save_file = os.path.join(save_dir, 'syllable_video.mp4')
        make_syllable_video(
            save_file, states, video_obj, markers=None, single_label=None,
            label_mapping=label_mapping, probs=probs, **video_kwargs)


class ReportGeneratorTraining(ReportGeneratorBase):
    """Generate a report after model training that includes plots and videos of labeled data.

    Examples
    --------
    # initialize object for training sessions
    reporter = GenerateReportTraining(
        model_dir="/path/to/model_directory",
    )
    # create report
    report_dir = reporter.generate_report(save_dir="/path/to/report_dir", format="pdf")

    Notes
    -----
    * csv files in features_directory must be named "<sess_id>_labeled.csv"
    * mp4 files in videos_directory must be names "<sess_id>.mp4"

    """

    def __init__(self, model_dir: str):
        """

        Parameters
        ----------
        model_dir
            directory that contains `best_val_model.pt` and `hparams.yaml` files

        """
        # store `model_dir` and load model
        super().__init__(model_dir=model_dir)

    def generate_report(
            self,
            sess_ids: list,
            data_dir: str,
            input_type: str,
            behaviors_to_keep: Optional[list] = None,
            video_framerate: Optional[int] = None,
            format: str = 'pdf',
            bout_example_kwargs: dict = {},
            **kwargs
    ):

        # collect info from each session
        metrics_df = []
        probs_pred_dict = {}
        states_pred_dict = {}
        states_hand_dict = {}
        features_dict = {}
        feature_names = None
        for sess_id in sess_ids:

            # initialize data handler; point to correct base path
            handler = DataHandler(sess_id, base_path=data_dir)
            if input_type == 'markers':
                handler.load_markers()
                features = handler.markers.vals
                feature_names = handler.markers.names
                input_file = handler.markers.path
            else:
                handler.load_features(dirname=input_type)
                features = handler.features.vals
                feature_names = handler.features.names
                input_file = handler.features.path

            # compute states from features
            probs_pred, states_pred = self.compute_states(sess_id, input_file)

            # load hand labels; update hand labels to only contain beh of interest
            handler.load_hand_labels()
            state_names = handler.hand_labels.names
            states_all = np.argmax(handler.hand_labels.vals, axis=1)
            if behaviors_to_keep is None:
                behaviors_to_keep = [l for l in state_names if l != 'background']
            cols_to_keep = [np.where(np.array(state_names) == b)[0][0] for b in behaviors_to_keep]
            states = np.zeros_like(states_all)
            for c, col_to_keep in enumerate(cols_to_keep):
                pos_idxs = np.where(states_all == col_to_keep)[0]
                states[pos_idxs] = c + 1
            cutoff = int(np.floor(states.shape[0] / self.hparams['sequence_length'])) \
                * self.hparams['sequence_length']
            states = states[:cutoff]

            # compute precision and recall
            scores = get_precision_recall(states, states_pred, background=None)

            # store info
            df_dict = {'sess_id': sess_id}
            for c, beh in enumerate(['background'] + behaviors_to_keep):
                df_dict['precision_%s' % beh] = scores['precision'][c]
                df_dict['recall_%s' % beh] = scores['recall'][c]
                df_dict['f1_%s' % beh] = scores['f1'][c]
                df_dict['rate_%s_hand' % beh] = len(np.where(states == c)[0]) / states.shape[0]
                df_dict['rate_%s_model' % beh] = \
                    len(np.where(states_pred == c)[0]) / states_pred.shape[0]
            metrics_df.append(pd.DataFrame(df_dict, index=[0]))

            probs_pred_dict[sess_id] = probs_pred
            states_pred_dict[sess_id] = states_pred
            states_hand_dict[sess_id] = states
            features_dict[sess_id] = features

        # combine all metrics into a single dataframe
        metrics_df = pd.concat(metrics_df)

        # generate diagnostic plots
        generate_training_report(
            metrics_df=metrics_df,
            probs_pred_dict=probs_pred_dict,
            states_pred_dict=states_pred_dict,
            states_hand_dict=states_hand_dict,
            state_names=self.state_names,
            features_dict=features_dict,
            feature_names=feature_names,
            train_metrics_file=os.path.join(self.model_dir, 'metrics.csv'),
            train_sess_ids=self.hparams['expt_ids'],
            video_framerate=video_framerate,
            save_dir=os.path.join(self.model_dir, 'diagnostics'),
            format=format,
            bout_example_kwargs=bout_example_kwargs,
            **kwargs
        )


def generate_training_report(
        metrics_df,
        probs_pred_dict,
        states_pred_dict,
        states_hand_dict,
        state_names,
        features_dict,
        feature_names,
        train_metrics_file,
        train_sess_ids,
        video_framerate,
        save_dir,
        format,
        bout_example_kwargs,
        **kwargs
):

    # parse inputs
    bout_example_kwargs_default = {
        'features_to_plot': None,
        'frame_win': 200,
        'max_n_ex': 10,
        'min_bout_len': 5,
    }
    bout_example_kwargs = update_kwargs_dict_with_defaults(
        bout_example_kwargs, bout_example_kwargs_default)

    os.makedirs(save_dir, exist_ok=True)

    # save out metrics df as a csv
    save_file = os.path.join(save_dir, 'model_metrics.csv')
    metrics_df.to_csv(save_file)

    # ------------------------------------------------
    # plots
    # ------------------------------------------------

    # training curves on train data
    save_file = os.path.join(save_dir, 'train_curves')
    plot_training_curves(
        train_metrics_file, dtype='train', expt_ids=train_sess_ids, save_file=save_file,
        format=format)

    # training curves on validation data
    save_file = os.path.join(save_dir, 'val_curves')
    plot_training_curves(
        train_metrics_file, dtype='val', expt_ids=train_sess_ids, save_file=save_file,
        format=format)

    # scatterplot of hand vs model rates
    save_file = os.path.join(save_dir, 'behavior_rate_scatters.%s' % format)
    title = None  # 'Session %s' % sess_id
    plot_rate_scatters(df=metrics_df, state_names=state_names, title=title, save_file=save_file)

    # plot examples of each behavior type (probs + markers + pred states + hand states)
    if bout_example_kwargs['features_to_plot'] is None:
        bout_example_kwargs['features_to_plot'] = feature_names
    idxs_features = np.where(
        np.isin(np.array(feature_names), np.array(bout_example_kwargs['features_to_plot'])))[0]
    for sess_id in features_dict.keys():
        features = features_dict[sess_id]
        features_ = features[:, idxs_features]
        states = states_pred_dict[sess_id]
        states_hand = states_hand_dict[sess_id]
        probs = probs_pred_dict[sess_id]
        bouts_w_idxs = get_label_runs([states_hand])
        for b, bouts_list in enumerate(bouts_w_idxs):
            state_name = state_names[b]
            save_file = os.path.join(
                save_dir, '%s_ethograms_%s_onset.%s' % (sess_id, state_name, format))
            f1 = metrics_df[metrics_df.sess_id == sess_id]['f1_%s' % state_name].values[0]
            title = '%s (%s onsets, F1=%1.2f)' % (sess_id, state_name, f1)
            plot_bout_onsets_w_features(
                bouts_list, markers=features_,
                marker_names=bout_example_kwargs['features_to_plot'],
                probs=probs, states=states, states_hand=states_hand, state_names=state_names,
                framerate=video_framerate, title=title, save_file=save_file, **bout_example_kwargs)
