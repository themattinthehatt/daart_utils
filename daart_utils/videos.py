"""Video making functions.

These functions create a temporary directory, save individual frames with matplotlib, then create
a video with ffmpeg.

"""

import matplotlib.pyplot as plt
import os
import shutil


def make_labeled_video(save_file, frames, points, framerate=20, height=4):
    """Behavioral video overlaid with markers.

    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    frames : np.ndarray
        array of shape (n_frames, n_channels, ypix, xpix)
    points : dict
        keys of marker names and vals of marker values, i.e. `points['paw_l'].shape = (n_t, 2)`
    framerate : float
        framerate of video
    height : float
        height of movie in inches

    """
    import cv2

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

    for n in range(n_frames):

        ax.clear()  # important!! otherwise each frame will plot on top of the last

        if n % 100 == 0:
            print('processing frame %03i/%03i' % (n, n_frames))

        # plot original frame
        ax.imshow(frames[n, 0], vmin=0, vmax=255, cmap='gray')
        # plot markers
        for m, (marker_name, marker_vals) in enumerate(points.items()):
            ax.plot(
                marker_vals[n, 0], marker_vals[n, 1], 'o', markersize=8)

        plt.savefig(os.path.join(tmp_dir, 'frame_%06i.jpeg' % n))

    save_video(save_file, tmp_dir, framerate)


def make_syllable_video():
    pass


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
