"""Video making functions.

These functions create a temporary directory, save individual frames with matplotlib, then create
a video with ffmpeg.

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from tqdm import tqdm

from daart_utils.utils import get_label_runs


def get_frames_from_idxs(cap, idxs):
    """Helper function to load video segments.

    Parameters
    ----------
    cap : cv2.VideoCapture object
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
            cap.set(1, i)
        ret, frame = cap.read()
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


def make_labeled_video(
        save_file, frames, frame_idxs=None, markers=None, probs=None, state_names=None,
        framerate=20, height=4, **kwargs):
    """Behavioral video overlaid with markers and discrete labels.

    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    frames : np.ndarray
        array of shape (n_frames, n_channels, ypix, xpix)
    frame_idxs : array-like
        frame index for each frame
    markers : dict, optional
        keys of marker names and vals of marker values, i.e. `markers[<bodypart>].shape = (n_t, 2)`
    probs : array-like
        shape (n_t, n_states)
    state_names : list
        name for each discrete behavioral state
    framerate : float, optional
        framerate of video
    height : float, optional
        height of movie in inches

    """

    tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    n_frames, _, img_height, img_width = frames.shape

    h = height
    w = h * (img_width / img_height)
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height, 0])
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

    txt_kwargs = {
        'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top', 'fontname': 'monospace', 'transform': ax.transAxes,
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none'),
    }

    txt_fr_kwargs = {
        'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'bottom', 'fontname': 'monospace', 'transform': ax.transAxes,
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none'),
    }

    for n in tqdm(range(n_frames)):

        ax.clear()  # important!! otherwise each frame will plot on top of the last

        # plot original frame
        ax.imshow(frames[n, 0], vmin=0, vmax=255, cmap='gray')

        # plot markers
        if markers is not None:
            for m, (marker_name, marker_vals) in enumerate(markers.items()):
                ax.plot(marker_vals[n, 0], marker_vals[n, 1], 'o', markersize=8)

        # annotate with labels
        if probs is not None and state_names is not None and frame_idxs is not None:
            # collect all labels present on this frame
            label_txt = ''
            for s, state_name in enumerate(state_names):
                label_txt += '%s: %1.2f' % (state_name, probs[frame_idxs[n], s])
                if s != len(state_names) - 1:
                    label_txt += '\n'
            # plot label string
            ax.text(0.02, 0.98, label_txt, **txt_kwargs)

        # add frame number
        if frame_idxs is not None:
            ax.text(0.02, 0.02, 'frame %i' % frame_idxs[n], **txt_fr_kwargs)

        plt.savefig(os.path.join(tmp_dir, 'frame_%06i.jpeg' % n))

    save_video(save_file, tmp_dir, framerate)


def make_syllable_video(
        save_file, labels, video_obj, markers=None, markersize=8, min_threshold=5, n_buffer=5,
        n_pre_frames=3, max_frames=1000, single_label=None, label_mapping=None, probs=None,
        framerate=20, **kwargs):
    """Composite video shows many clips belonging to same behavioral class, one panel per class.

    Adapted from:
    https://github.com/themattinthehatt/behavenet/blob/master/behavenet/plotting/arhmm_utils.py#L360

    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    labels : array-like
        discrete labels for each time points, shape (n_t,)
    video_obj : daart_utils.data.Video object
        contains function to load frames
    markers : dict, optional
        keys of marker names and vals of marker values, i.e. `markers[<bodypart>].shape = (n_t, 2)`
    markersize : float, optional
        size of markers if plotted
    min_threshold : int, optional
        minimum length of syllable clips
    n_buffer : int, optional
        number of black frames between clips
    n_pre_frames : int, optional
        nunber of frames before syllable onset
    max_frames : int, optional
        length of video
    single_label : int, optional
        choose only a single label for movie; if NoneType, all labels included
    label_mapping : dict
        mapping from label number to label name
    probs : array-like
        probability of each label
    framerate : float, optional
        framerate of video

    """

    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter

    frame_idxs = [np.arange(len(labels))]
    xpix = video_obj.frame_width
    ypix = video_obj.frame_height

    # separate labels
    if not isinstance(labels, list):
        labels = [labels]
    label_idxs = get_label_runs(labels)
    K = len(label_idxs)

    # get all example over threshold
    np.random.seed(0)
    labels_list = [[] for _ in range(K)]
    for curr_label in range(K):
        if label_idxs[curr_label].shape[0] > 0:
            # grab all good bouts
            good_bouts_idxs = np.where(
                np.diff(label_idxs[curr_label][:, 1:3], 1) > min_threshold)[0]
            # randomize
            np.random.shuffle(good_bouts_idxs)
            labels_list[curr_label] = label_idxs[curr_label][good_bouts_idxs]

    if single_label is not None:
        K = 1
        fig_width = 3
    else:
        fig_width = 10
    n_rows = int(np.round(np.sqrt(K)))
    n_cols = int(np.ceil(K / n_rows))

    # initialize syllable movie frames
    plt.clf()
    fig_dim_div = xpix * n_cols / fig_width
    fig_width = (xpix * n_cols) / fig_dim_div
    fig_height = (ypix * n_rows) / fig_dim_div
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    for i, ax in enumerate(fig.axes):
        ax.set_yticks([])
        ax.set_xticks([])
        if i >= K:
            ax.set_axis_off()
        # elif single_label is not None:
        #     ax.set_title('Syllable %i' % single_label, fontsize=16)
        # else:
        #     ax.set_title('Syllable %i' % i, fontsize=16)
    # fig.tight_layout(pad=0, h_pad=1)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

    # hard coded params
    im_kwargs = {'animated': True, 'vmin': 0, 'vmax': 255, 'cmap': 'gray'}

    txt_kwargs = {
        'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top', 'fontname': 'monospace',
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none')}
    txt_offset_x = 10
    txt_offset_y = 5

    txt_fr_kwargs = {
        'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'bottom', 'fontname': 'monospace',
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none')}
    txt_offset_x_fr = 10
    txt_offset_y_fr = ypix - 5

    # loop through syllables
    ims = [[] for _ in range(max_frames)]
    for i_k, ax in tqdm(enumerate(fig.axes)):

        # plot black frames if no syllable in this axis
        if i_k >= K:
            for i_frame in range(max_frames):
                im = ax.imshow(np.zeros((ypix, xpix)), **im_kwargs)
                ims[i_frame].append(im)
            continue

        # select correct label index
        if single_label is not None:
            i_k = single_label

        # plot black frame if no frames exist for this label
        if len(labels_list[i_k]) == 0:
            for i_frame in range(max_frames):
                im = ax.imshow(np.zeros((ypix, xpix)), **im_kwargs)
                ims[i_frame].append(im)
            continue

        # get text for this label
        if K < 10:
            label_txt = '%i' % i_k if label_mapping is None else label_mapping[i_k]
        else:
            label_txt = '%02i' % i_k if label_mapping is None else label_mapping[i_k]

        i_clip = 0  # keep track of clip number
        i_frame = 0  # keep track of overall number of frames
        while i_frame < max_frames:

            if i_clip >= len(labels_list[i_k]):
                if single_label is not None:
                    # no more plotting
                    break
                else:
                    # plot black
                    im = ax.imshow(np.zeros((ypix, xpix)), **im_kwargs)
                    ims[i_frame].append(im)
                    i_frame += 1
            else:
                # get indices into clip
                i_idx = labels_list[i_k][i_clip, 0]
                i_beg = labels_list[i_k][i_clip, 1]
                i_end = labels_list[i_k][i_clip, 2]
                # use these to get indices into frames
                m_beg = frame_idxs[i_idx][max(0, i_beg - n_pre_frames)]
                m_end = frame_idxs[i_idx][i_end]
                # grab movie clip
                movie_clip = video_obj.get_frames_from_idxs(np.arange(m_beg, m_end))[:, 0, :, :]

                # basic error check
                i_non_k = labels[i_idx][i_beg:i_end] != i_k
                if np.any(i_non_k):
                    raise ValueError('Misaligned labels for syllable segmentation')

                # loop over this clip
                for i in range(movie_clip.shape[0]):

                    # in case clip is too long
                    if i_frame >= max_frames:
                        continue

                    # display frame
                    im = ax.imshow(movie_clip[i], **im_kwargs)
                    ims[i_frame].append(im)

                    # marker overlay
                    if markers is not None:
                        n = m_beg + i  # absolute index
                        ax.set_prop_cycle(None)  # reset color cycle
                        for m, (marker_name, marker_vals) in enumerate(markers.items()):
                            im = ax.plot(
                                marker_vals[n, 0], marker_vals[n, 1], 'o',
                                markersize=markersize)[0]
                            ims[i_frame].append(im)

                    # text on top: state
                    if probs is not None:
                        label_txt_tmp = '%s: %1.2f' % (label_txt, probs[m_beg + i, i_k])
                    else:
                        label_txt_tmp = label_txt
                    im = ax.text(txt_offset_x, txt_offset_y, label_txt_tmp, **txt_kwargs)
                    ims[i_frame].append(im)

                    # text on bottom: frame
                    frame_txt = 'frame %i' % (m_beg + i)
                    im = ax.text(txt_offset_x_fr, txt_offset_y_fr, frame_txt, **txt_fr_kwargs)
                    ims[i_frame].append(im)

                    i_frame += 1

                # add buffer black frames
                for j in range(n_buffer):
                    # in case chunk is too long
                    if i_frame >= max_frames:
                        continue
                    im = ax.imshow(np.zeros((ypix, xpix)), **im_kwargs)
                    ims[i_frame].append(im)
                    if single_label is None:
                        im = ax.text(txt_offset_x, txt_offset_y, label_txt, **txt_kwargs)
                        ims[i_frame].append(im)
                    i_frame += 1

                i_clip += 1

    print('creating animation...', end='')
    ani = animation.ArtistAnimation(
        fig, [ims[i] for i in range(len(ims)) if ims[i] != []], blit=True, repeat=False)
    print('done')
    print('saving video to %s...' % save_file, end='')
    writer = FFMpegWriter(fps=framerate, bitrate=-1)
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    ani.save(save_file, writer=writer)
    print('done')


def save_video(save_file, tmp_dir, framerate=20):
    """Create video from temporary directory of images; will delete images and temporary directory.

    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    tmp_dir : str
        temporary directory that stores frames of video; this directory will be deleted
    framerate : float, optional
        framerate of final video

    """
    import subprocess

    if os.path.exists(save_file):
        os.remove(save_file)

    # make mp4 from images using ffmpeg
    call_str = \
        'ffmpeg -r %f -i %s -c:v libx264 %s' % (
            framerate, os.path.join(tmp_dir, 'frame_%06d.jpeg'), save_file)
    print(call_str)
    subprocess.run(['/bin/bash', '-c', call_str], check=True)

    # delete tmp directory
    shutil.rmtree(tmp_dir)
