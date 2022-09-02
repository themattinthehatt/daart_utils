"""Analyze predictions on unlabeled video data.

Users select an arbitrary number of feature csvs (one per session) from their file system

The app creates plots for:
- histograms of behavioral bout durations
- distribution of behaviors
- sample snippets of model probabilities and features for each behavior
- labeled video
- syllable video

to run from command line:
> streamlit run /path/to/diagnostics.py

optionally, multiple feature files can be specified from the command line; each must be
preceded by "--feature_files":
> streamlit run /path/to/diagnostics.py --
--feature_files=/path/to/sess0_labeled.csv --feature_files=/path/to/sess1_labeled.csv

Notes:
    - this file should only contain the streamlit logic for the user interface
    - data processing should come from (cached) functions imported from daart_utils.reports
    - plots should come from (non-cached) functions imported from daart_utils.plotting

"""

import argparse
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import streamlit as st
import yaml

# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

from daart.transforms import MakeOneHot

from daart_utils.data import DataHandler
from daart_utils.plotting import plotly_markers_and_states
from daart_utils.reports import ReportGenerator
from daart_utils.streamlit_utils import update_single_file, update_file_list

if 'handler' not in st.session_state:
    st.session_state.handler = None
if 'fig_traces' not in st.session_state:
    st.session_state.fig_traces = None
if 'idx' not in st.session_state:
    st.session_state.idx = int(0)
if 'window' not in st.session_state:
    st.session_state.window = int(200)
if 'frame_skip' not in st.session_state:
    st.session_state.frame_skip = int(10)
if 'include_video' not in st.session_state:
    st.session_state.include_video = False
if 'update_plot' not in st.session_state:
    st.session_state.update_plot = False


def st_directory_picker(initial_path=Path()):
    """
    adapted from
    https://github.com/aidanjungo/StreamlitDirectoryPicker/blob/main/directorypicker.py
    """

    if "path" not in st.session_state:
        st.session_state.path = initial_path.absolute()

    st.text_input("Selected directory:", st.session_state.path)

    _, col1, col2, col3, _ = st.columns([3, 1, 3, 1, 3])

    with col1:
        st.markdown('#')
        if st.button('⬅️', key='dir_back') and 'path' in st.session_state:
            st.session_state.path = st.session_state.path.parent
            st.experimental_rerun()

    with col2:
        subdirectroies = [
            f.stem
            for f in st.session_state.path.iterdir()
            if f.is_dir()
            and (not f.stem.startswith(".") and not f.stem.startswith("__"))
        ]
        if subdirectroies:
            st.session_state.new_dir = st.selectbox(
                'Subdirectories', sorted(subdirectroies)
            )
        else:
            st.markdown("#")
            st.markdown(
                "<font color='#FF0000'>No subdir</font>", unsafe_allow_html=True
            )

    with col3:
        if subdirectroies:
            st.markdown('#')
            if st.button('➡️', key='dir_forward') and 'path' in st.session_state:
                st.session_state.path = Path(
                    st.session_state.path, st.session_state.new_dir
                )
                st.experimental_rerun()

    return st.session_state.path


def zscore(array):
    array_centered = array - np.mean(array, axis=0)
    array_z = array_centered / np.std(array_centered, axis=0)
    return array_z


def bound_value(val, min_val, max_val):
    return max(min(val, max_val), min_val)


@st.cache
def get_sess_ids(sess_dir):
    sess_ids = ReportGenerator.find_session_ids(sess_dir)
    sess_ids.sort()
    return sess_ids


@st.cache(hash_funcs={DataHandler: lambda _: None})
def load_data(sess_id, data_dir, trace_dir):
    # init data handler for easy data loading/manipulation
    handler = DataHandler(session_id=sess_id, base_path=data_dir)
    # load traces, minimum required data
    handler.load_features(dirname=trace_dir)
    handler.features.vals = zscore(handler.features.vals)
    # load model labels
    handler.load_model_labels(logits=True)
    # load hand labels
    if os.path.exists(os.path.join(data_dir, 'labels-hand')):
        handler.load_hand_labels()
    # load video
    if os.path.exists(os.path.join(data_dir, 'videos')):
        handler.load_video()
    return handler


def run():

    # args = parser.parse_args()

    st.title('Segmentation Diagnostics')

    # ---------------------------------------------------------------------------------------------
    # load data
    # ---------------------------------------------------------------------------------------------
    st.markdown('#### Select data directory')

    with st.expander('Expand for data directory formatting info'):

        st.markdown("""
            The data directory should contain a set of subdirectories for different data types
            (markers, states, videos, etc.).
            The required format is the following (with data from two example sessions):

            ```
            data_directory
            ├── features
            │   ├── <sess_id_0>_labeled.csv
            │   └── <sess_id_1>_labeled.csv
            ├── labels-hand
            │   ├── <sess_id_0>_labels.csv
            │   └── <sess_id_1>_labels.csv
            ├── labels-model
            │   ├── <sess_id_0>_states.csv
            │   └── <sess_id_1>_states.csv
            └── videos
                ├── <sess_id_0>.mp4
                └── <sess_id_1>.mp4
            ```

            * `features` contains traces used to fit the models; these could be raw markers or some
            other type of feature; you will be able to specify the name of this directory below
            * `labels-hand` is optional and contains the ground truth hand labels; if this
            directory is present these data will be plotted along with model predictions
            * `labels-model` contains the state probabilities for each frame
            * `videos` is optional and contains mp4 files for each session that can be plotted
            along with traces and predicted states

        """)

    trace_dir = st.text_input(
        'Directory name for traces (e.g. `markers` or `features`)',
        value='features')
    model_labels_dir = 'labels-model'
    data_dir = st_directory_picker()

    # trace_dir = 'features-aug'
    # data_dir = '/media/mattw/behavior/daart-data/fish/tmp'

    available_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if trace_dir not in available_dirs or model_labels_dir not in available_dirs:
        sess_ids = []
        st.warning('Current directory does not contain the correct subdirectories')
    else:
        sess_ids = get_sess_ids(os.path.join(data_dir, trace_dir))

    if len(sess_ids) > 0:

        sess_id = st.selectbox('Select a session to load', sess_ids)

        load_data_submit = st.button('Load data')
        if load_data_submit:
            st.session_state.handler = load_data(sess_id, data_dir, trace_dir)

    # ---------------------------------------------------------------------------------------------
    # plot options
    # ---------------------------------------------------------------------------------------------
    if st.session_state.handler is not None:

        st.text('')
        st.text('')
        st.markdown('#### User options')

        # select which features to plot
        st.markdown('###### Plotting options')
        with st.form('plot_options'):

            with st.expander('Select features to plot'):
                include_feature = {f: True for f in st.session_state.handler.features.names}

                for f in st.session_state.handler.features.names:
                    include_feature[f] = st.checkbox(f, value=include_feature[f])

            if st.session_state.include_video:
                window_ = st.text_input('Plot width (frames)', st.session_state.window)
                window_ = int(window_)
                if st.session_state.window != window_:
                    st.session_state.window = window_

            update_plot = st.form_submit_button('Update plot')
            # signal update if True; if False, leave alone since an update might be triggered
            # elsewhere
            if update_plot:
                st.session_state.update_plot = True

        # select video options
        if st.session_state.handler.video is not None:
            st.markdown('###### Video options')

            include_video = st.checkbox('Include video data')
            if include_video != st.session_state.include_video:
                st.session_state.update_plot = True  # update plot when adding/removing video
                st.session_state.include_video = include_video
                st.experimental_rerun()

            if st.session_state.include_video:
                skip_ = st.text_input('Arrow skip (frames)', st.session_state.frame_skip)
                skip_ = int(skip_)
                if st.session_state.frame_skip != skip_:
                    st.session_state.frame_skip = skip_

        else:
            st.session_state.include_video = False

    # text
    if st.session_state.fig_traces is not None or st.session_state.update_plot:
        st.text('')
        st.text('')
        st.markdown('#### Data UI')
        st.text('')

    # ---------------------------------------------------------------------------------------------
    # plot frames
    # ---------------------------------------------------------------------------------------------
    if st.session_state.handler is not None and st.session_state.include_video:

        min_frames = 0
        max_frames = int(st.session_state.handler.video.n_frames - 1)

        # select frame index
        col0, col1, col2, col3, col4 = st.columns([1, 1, 6, 1, 1])
        with col0:
            st.markdown('#')
            if st.button('️⬅️⬅️', key='frame_back_n'):
                st.session_state.idx -= st.session_state.frame_skip
                st.session_state.update_plot = True
                st.experimental_rerun()  # update all input elements
        with col1:
            st.markdown('#')
            if st.button('️⬅️', key='frame_back_1'):
                st.session_state.idx -= 1
                st.session_state.update_plot = True
                st.experimental_rerun()  # update all input elements
        with col2:
            idx_ = st.slider(
                'Select frame index',
                min_value=min_frames, max_value=max_frames,
                value=st.session_state.idx
            )
            if st.session_state.idx != int(idx_):
                st.session_state.idx = int(idx_)
                st.session_state.update_plot = True
                st.experimental_rerun()  # update all input elements
        with col3:
            st.markdown('#')
            if st.button('➡️', key='frame_forward_1'):
                st.session_state.idx += 1
                st.session_state.update_plot = True
                st.experimental_rerun()  # update all input elements
        with col4:
            st.markdown('#')
            if st.button('➡️➡️', key='frame_forward_n'):
                st.session_state.idx += st.session_state.frame_skip
                st.session_state.update_plot = True
                st.experimental_rerun()  # update all input elements

        # safeguard from accessing bad frames
        st.session_state.idx = bound_value(st.session_state.idx, min_frames, max_frames)

        # plot
        st.session_state.handler.video.cap.set(1, st.session_state.idx)
        ret, frame = st.session_state.handler.video.cap.read()
        if not ret:
            raise Exception

        # st.columns is a hacky way to center the video frame
        _, col0v, col1v, col2v, _ = st.columns([1, 1, 3, 1, 1])
        with col0v:
            st.write('')
        with col1v:
            st.image(frame, channels="BGR", width=300)
        with col2v:
            st.write('')

    # ---------------------------------------------------------------------------------------------
    # plot traces
    # ---------------------------------------------------------------------------------------------
    if st.session_state.handler is not None:

        if st.session_state.update_plot:
            state_names = (st.session_state.handler.hand_labels.names
                           or st.session_state.handler.model_labels.names)

            # x-axis
            if st.session_state.include_video:
                x = np.arange(
                    int(st.session_state.idx - st.session_state.window),
                    min(
                        int(st.session_state.idx + st.session_state.window),
                        int(st.session_state.handler.video.n_frames - 1)),
                )
                add_vertical_line = st.session_state.idx
            else:
                x = np.arange(st.session_state.handler.model_labels.vals.shape[0])
                add_vertical_line = None

            # hand labels
            if st.session_state.handler.hand_labels.vals is not None:
                states_hand = st.session_state.handler.hand_labels.vals[x]
                states_hand[states_hand != 1] = np.nan
            else:
                states_hand = None

            # model outputs
            states_ = np.argmax(st.session_state.handler.model_labels.vals[x], axis=1)
            states = MakeOneHot(n_classes=len(state_names))(states_)
            states[states != 1] = np.nan

            state_probs = st.session_state.handler.model_labels.vals[x]
            features = st.session_state.handler.features.vals[x]

            # crop beginning if necessary
            if st.session_state.include_video \
                    and (st.session_state.idx - st.session_state.window) < 0:
                d = st.session_state.window - st.session_state.idx
                if states_hand is not None:
                    states_hand[:d] = np.nan
                states[:d] = np.nan
                state_probs[:d] = np.nan
                features[:d] = np.nan

            fig_traces = plotly_markers_and_states(
                x=x,
                states_hand=states_hand,
                states_model=states,
                states_probs=state_probs,
                state_names=state_names,
                features=features,
                include_feature=include_feature,
                add_vertical_line=add_vertical_line,
            )

            st.session_state.fig_traces = fig_traces
            st.session_state.update_plot = False

        if st.session_state.fig_traces is not None:
            # st.session_state.fig_traces.update_xaxes(
            #     spikemode='across', spikedash='solid', spikecolor='black', spikethickness=0.5)
            # st.session_state.fig_traces.update_traces(xaxis='x%i' % st.session_state.n_rows)
            st.plotly_chart(st.session_state.fig_traces)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--feature_files', action='append', default=[])

    run()
