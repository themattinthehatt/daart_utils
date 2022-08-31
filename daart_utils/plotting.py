"""Plotting functions for daart models."""

import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

from daart.eval import run_lengths


def plot_heatmaps(
        df, metric, sess_ids, title='', kind=1, n_cols=2, center=False, vmin=0, vmax=1, agg='mean',
        annot=False, cmaps=None, save_file=None, **kwargs):
    """Plot a heatmap of model performance as a function of hyperparameters

    Parameters
    ----------
    df : pd.DataFrame
        must include the following columns: label, lambda_weak, lambda_pred, `metric`, expt_id
    metric : str
        metric whose values will be displayed with heatmap; column name in `df`
    sess_ids : list
    title : str
    kind : int
        1: plot heatmap for each label, expt id
        2: plot heatmap for each label, averaged over expt ids
        3: plot single heatmap, average over labels and expt ids
    n_cols : int
        for kind=1 and kind=2, a plot is created for each label name; n_cols sets number of columns
        in this array
    center : int
        True to center heatmap at 0, False to set plot limits as [0, 1]
    vmin : float
    vmax : float
    agg : str
        'mean' | 'median' | 'var'
        method used to aggregate data across sessions when kind=2 or kind=3
    annot : bool
    cmaps : list of strs
    save_file : str

    """

    label_names = df.label.unique()

    n_rows = int(np.ceil(len(label_names) / n_cols))

    if kind == 1:

        # plot heatmap for each label, expt id
        for sess_id in sess_ids:

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 1, 4 * n_rows + 1))
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]
            for ax in axes:
                ax.set_axis_off()

            for l, label_name in enumerate(label_names):
                axes[l].set_axis_on()
                data_tmp = df[(df.label == label_name) & (df.expt_id == sess_id)].pivot(
                    'lambda_weak', 'lambda_pred', metric)
                sns.heatmap(data_tmp, vmin=vmin, vmax=vmax, ax=axes[l], annot=annot, **kwargs)
                axes[l].invert_yaxis()
                axes[l].set_title(label_name)
            fig.suptitle('%s %s' % (sess_id, title))
            plt.tight_layout()
            plt.show()

    elif kind == 2:

        # plot heatmap for each label, averaged over expt ids
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 1, 4 * n_rows + 1))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        for ax in axes:
            ax.set_axis_off()

        for l, label_name in enumerate(label_names):

            if agg == 'mean':
                data_tmp = df[(df.label == label_name)].groupby(
                    ['lambda_weak', 'lambda_pred']).mean().reset_index().pivot(
                    'lambda_weak', 'lambda_pred', metric)
            elif agg == 'median':
                data_tmp = df[(df.label == label_name)].groupby(
                    ['lambda_weak', 'lambda_pred']).median().reset_index().pivot(
                    'lambda_weak', 'lambda_pred', metric)
            elif agg == 'var':
                data_tmp = df[(df.label == label_name)].groupby(
                    ['lambda_weak', 'lambda_pred']).var().reset_index().pivot(
                    'lambda_weak', 'lambda_pred', metric)
            else:
                raise NotImplementedError

            axes[l].set_axis_on()
            if cmaps is not None:
                sns.heatmap(
                    data_tmp, vmin=vmin, vmax=vmax, ax=axes[l], annot=annot, cmap=cmaps[l],
                    **kwargs)
            elif center:
                sns.heatmap(data_tmp, center=0, ax=axes[l], annot=annot, **kwargs)
            else:
                sns.heatmap(data_tmp, vmin=vmin, vmax=vmax, ax=axes[l], annot=annot, **kwargs)
            axes[l].invert_yaxis()
            axes[l].set_title(label_name)

        fig.suptitle(title)
        plt.tight_layout()
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()

    elif kind == 3:

        # plot single heatmap, average over labels and expt ids
        fig = plt.figure(figsize=(5, 4))

        if agg == 'mean':
            data_tmp = df.groupby(
                ['lambda_weak', 'lambda_pred']).mean().reset_index().pivot(
                'lambda_weak', 'lambda_pred', metric)
        elif agg == 'median':
            data_tmp = df.groupby(
                ['lambda_weak', 'lambda_pred']).median().reset_index().pivot(
                'lambda_weak', 'lambda_pred', metric)
        elif agg == 'var':
            data_tmp = df.groupby(
                ['lambda_weak', 'lambda_pred']).var().reset_index().pivot(
                'lambda_weak', 'lambda_pred', metric)
        else:
            raise NotImplementedError

        if center:
            sns.heatmap(data_tmp, center=0, annot=annot, **kwargs)
        else:
            sns.heatmap(data_tmp, vmin=vmin, vmax=vmax, annot=annot, **kwargs)
        fig.gca().invert_yaxis()

        fig.suptitle(title)
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()


def plot_bout_histograms(
        bouts, state_names=None, framerate=None, n_cols=3, title=None, save_file=None):
    """Plot histogram of bout durations for each behavior.

    Parameters
    ----------
    bouts : dict
        output of daart.eval.run_lengths
        keys are behavior numbers, vals are lists of bout durations
    state_names : list, optional
    framerate : float, optional
        framerate of the original video; if None, bout durations are reported in frames, if float
        then seconds are used
    n_cols : int
        number of columns in the plot
    title : str, optional
        figure title
    save_file : str, optional
        absolute path for saving (including extension)

    """

    if not state_names:
        state_names = ['class_%i' % i for i in range(len(bouts))]

    if framerate is not None:
        bouts_ = {k: np.array(v) / framerate for k, v in bouts.items()}
        norm_type = 'sec'
    else:
        bouts_ = bouts.copy()
        norm_type = 'frames'

    n_rows = int(np.ceil(len(state_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for a, ax in enumerate(axes):
        if a < len(bouts):
            m = np.ceil(np.log10(np.max(bouts_[a])))
            bins = 10 ** (np.arange(0, m, m / 20))
            ax.hist(
                bouts_[a],
                #             density=True,
                alpha=0.8,
                bins=bins,
            )
            ax.set_xscale('log')
            ticks = np.around(10 ** np.arange(0, m, m / 4), decimals=2)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks)
            if a % n_cols == 0:
                # left edge
                ax.set_ylabel('Count')
            if np.floor(a / n_cols) + 1 == n_rows:
                ax.set_xlabel('Bout duration (%s)' % norm_type)
            ax.set_title('%s (%i bouts)\nmin=%i, max=%i' %
                         (state_names[a], len(bouts_[a]), np.min(bouts_[a]), np.max(bouts_[a])))
        else:
            ax.set_axis_off()

    if n_rows == 1:
        plt.subplots_adjust(top=0.75)
    elif n_rows == 2:
        plt.subplots_adjust(top=0.9)
    if title is None:
        title = 'Bout histograms'
    plt.suptitle(title)
    plt.tight_layout()

    if save_file:
        plt.savefig(save_file)

    plt.show()


def compute_transition_matrix(states, n_states=None):

    if n_states is None:
        n_states = len(np.unique(states))

    # get array of bout ids rather than frame-wise behavior ids
    # i.e. [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2] -> [0, 1, 0, 2]
    idxs_change = np.diff(states)
    bout_ids = states[np.where(idxs_change != 0)[0]]

    trans_mat = np.nan * np.zeros((n_states, n_states))
    for n in range(n_states):
        idxs_tmp = np.where(bout_ids == n)[0]
        next_bout = bout_ids[idxs_tmp[:-1] + 1]
        for m in range(n_states):
            if n == m:
                continue
            else:
                # find next state
                trans_mat[n, m] = len(np.where(next_bout == m)[0]) / len(next_bout)

    return trans_mat


def plot_behavior_distribution(
        states, state_names=None, framerate=None, title=None, save_file=None, **kwargs):
    """Plot a single bar plot of total behavior durations.

    Parameters
    ----------
    states : array-like
    state_names : list, optional
    framerate : float, optional
        framerate of the original video; if None, bout durations are reported in frames, if float
        then seconds are used
    title : str, optional
        figure title
    save_file : str, optional
        absolute path for saving (including extension)

    """

    if state_names is not None:
        n_states = len(state_names)
    else:
        n_states = len(np.unique(states))

    bouts = run_lengths(states)
    trans_mat = compute_transition_matrix(states, n_states=n_states)

    if not state_names:
        state_names = ['class_%i' % i for i in range(n_states)]
    state_names = [c.capitalize() for c in state_names]

    if framerate is not None:
        bouts_ = {k: np.array(v) / framerate for k, v in bouts.items()}
        norm_type = 'sec'
    else:
        bouts_ = bouts.copy()
        norm_type = 'frames'

    dist_x = []
    dist_y = []
    for b, bouts_list in bouts_.items():
        dist_x.append(b)
        dist_y.append(np.sum(bouts_list))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # bar plot of behaviors
    ax = axes[0]
    ax.bar(dist_x, dist_y, tick_label=state_names)
    ax.set_xlabel('Behavior', fontsize=12)
    ax.set_ylabel('Duration (%s)' % norm_type, fontsize=12)
    ax.set_title('Behavior distribution', fontsize=14)

    # bout transition matrix
    ax = axes[1]
    cmap = copy.copy(mpl.cm.get_cmap("Blues_r"))
    cmap.set_bad('black', 1.0)
    im = ax.imshow(trans_mat, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_states))
    ax.set_xticklabels(state_names)
    ax.set_xlabel('State n+1', fontsize=12)
    ax.set_yticks(np.arange(n_states))
    ax.set_yticklabels(state_names)
    ax.set_ylabel('State n', fontsize=12)
    for (j, i), label in np.ndenumerate(trans_mat):
        if i == j:
            continue
        if label < 0.4:
            fontcolor = 'white'
        else:
            fontcolor = 'black'
        ax.text(
            i, j, np.around(label, 2),
            ha='center', va='center',
            color=fontcolor, fontsize=12)
    ax.set_title('Empirical bout transition probabilities', fontsize=14)
    # fig.subplots_adjust(right=0.8)
    # # lower left corner in [0.83, 0.25]
    # # axes width 0.02 and height 0.8
    # cb_ax = fig.add_axes([0.83, 0.25, 0.04, 0.5])
    # fig.colorbar(im, cax=cb_ax)

    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file)

    # plt.show()


def plot_rate_scatters(df, state_names, title=None, save_file=None, **kwargs):

    n_states = len(state_names)
    n_cols = 2
    n_rows = int(np.ceil(n_states / n_cols))

    fig, axes = plt.subplots(n_cols, n_rows, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for ax in axes:
        ax.set_axis_off()

    for a, label in enumerate(state_names):

        ax = axes[a]
        ax.set_axis_on()

        # plot points
        x = 'rate_%s_hand' % label
        y = 'rate_%s_model' % label
        ax = sns.regplot(x=x, y=y, data=df, fit_reg=True, ax=ax)

        # plot line of best fit
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df[x], df[y])
        mn = 0.95 * np.min([df[x].min(), df[y].min()])
        mx = 1.05 * np.max([df[x].max(), df[y].max()])
        ax.plot([mn, mx], [mn, mx], '--', color='k')
        ax.text(
            0.7, 0.3, 'slope: %1.2f' % slope, ha='left', va='bottom', fontsize=14,
            transform=ax.transAxes)
        ax.text(
            0.7, 0.2, 'r: %1.2f' % r_value, ha='left', va='bottom', fontsize=14,
            transform=ax.transAxes)
        ax.text(
            0.7, 0.1, 'N: %i' % df.shape[0], ha='left', va='bottom', fontsize=14,
            transform=ax.transAxes)

        ax.set_xlabel('%s rate (hand)' % label.capitalize())
        ax.set_ylabel('%s rate (model)' % label.capitalize())

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file)

    # plt.show()


def get_state_colors(n_colors=6):
    """

    Parameters
    ----------
    n_colors : int
        total number of colors to use in colormap (number of states)

    Returns
    -------
    matplotlib.cm.ListedColormap

    """
    import matplotlib.cm as cm
    tmp = cm.get_cmap('tab10', 10)  # cm.get_cmap('Accent', 6)
    cmap = cm.get_cmap('tab10', n_colors)
    cmap.colors[0, :] = [0, 0, 0, .4]
    cmap.colors[1:, :] = tmp.colors[:n_colors - 1]
    return cmap


def plot_markers_and_states(
        times, markers, states, title, ax, cmap, ymin=None, ymax=None, spc=None,
        marker_names=None, **kwargs):
    """Plot markers or features with background color denoting discrete state.

    Parameters
    ----------
    times : array-like
        x-values of markers, shape (n_t,)
    markers : array-like
        shape (n_t, n_markers/n_features)
    states : array-like
        dense representation of discrete states, shape (n_t,)
    title : str or NoneType
        axis title
    ax : matplotlib.axes.Axes object
        axis in which to plot the markers and states
    cmap : matplotlib.cm.ListedColormap
        colormap for discrete states
    ymin : float, optional
    ymax : float, optional
    spc : float, optional
        control spacing between markers/features
    marker_names : list, optional
        marker names for y-axis labels

    """

    if ymin is None:
        ymin = np.min(markers)
    if ymax is None:
        ymax = np.max(markers)
    if spc is None:
        raise NotImplementedError
    n_frames = states.shape[0]
    n_colors = cmap.colors.shape[0]
    states_aug = np.concatenate([states[None, :], np.array([[0, n_colors - 1]])], axis=1)
    im = ax.imshow(
        states_aug, aspect='auto',
        extent=(0, n_frames - 1, ymin, ymax), interpolation='none',
        cmap=cmap)
    # plot markers
    n_markers = markers.shape[1]
    for n in range(n_markers):
        ax.plot(times, markers[:, n], color='k', linewidth=1)
    ax.set_xlim([0, n_frames - 1])
    if marker_names is not None:
        ax.set_yticks(spc * np.arange(len(marker_names)))
        ax.set_yticklabels(marker_names, fontsize=10)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    if title is not None:
        ax.set_title(title)
    return im


def plot_bout_onsets_w_features(
        bouts, markers, marker_names, probs, states, state_names, states_hand=None,
        frame_win=200, framerate=1, max_n_ex=10, min_bout_len=5, title=None, save_file=None,
        **kwargs
):
    """Plot array of behavior bout examples aligned to bout onset; each example is markers+states.

    Parameters
    ----------
    bouts : array-like
        shape (n_bouts, 3) where the 3 cols are (chunk_idx, idx_start, idx_end). `chunk_idx`
        denotes the index of the original list of state arrays that the current bout is found in,
        `idx_start` and `idx_end` denote the start/end indices of the current bout with respect to
        the chunk.
    markers : array-like
        shape (n_t, n_markers)
    marker_names : list
        name for each column in `markers`
    probs : array-like
        shape (n_t, n_states)
    states : array-like
        shape (n_t,)
    state_names : list
        name for each discrete behavioral state
    states_hand : array-like
        shape (n_t,); ground truth hand labels that will be added to plots if present
    frame_win : int, optional
        number of frames before/after bout onset to plot
    framerate : int, optional
        framerate of the original video; if None, bout durations are reported in frames, if float
        then seconds are used
    max_n_ex : int, optional
        max number of bout examples to plot
    min_bout_len : int, optional
        minimum length of bout required for plotting example
    title : str, optional
        figure title
    save_file : str, optional
        absolute path for saving (including extension)

    """

    sns.set_context('paper')
    sns.set_style('white')

    n_colors = len(state_names)
    cmap = get_state_colors(n_colors=n_colors)

    n_markers = markers.shape[1]
    markers_z = (markers - np.mean(markers, axis=0)) / np.std(markers, axis=0)
    spc = 0.2 * abs(markers_z.max())
    plotting_markers = markers_z + spc * np.arange(n_markers)
    ymin = min(-spc - 0.5, np.percentile(plotting_markers, 2))
    ymax = max(spc * n_markers, np.percentile(plotting_markers, 98))

    # get rid of bouts that fall outside of our window
    bouts_clean = bouts[(bouts[:, 1] > frame_win) & (bouts[:, 2] < (states.shape[0] - frame_win))]
    # get rid of short bouts
    bout_lens = bouts_clean[:, 2] - bouts_clean[:, 1]
    bouts_clean = bouts_clean[bout_lens >= min_bout_len]

    n_ex = min(bouts_clean.shape[0], max_n_ex)
    n_cols = 2
    n_rows = int(np.ceil(n_ex / n_cols))

    n_secs = np.floor(frame_win / framerate) - 1
    xticklabels = np.array([-n_secs, -n_secs / 2, 0, n_secs / 2, n_secs])
    xticks = xticklabels * framerate + frame_win

    if n_ex < 2:
        print('Did not find enough behavioral bouts for plotting')
        return

    if n_ex < 6:
        offset = 1
    else:
        offset = 0
    fig_width = 8
    fig_height = 2.5 * np.ceil(n_ex / 2) + offset
    plt.cla()
    plt.clf()
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    outer_grid = fig.add_gridspec(int(np.ceil(n_ex / 2)), 2, hspace=0.2)
    if states_hand is not None:
        inner_grid = [
            outer_grid[r].subgridspec(4, 1, hspace=0, height_ratios=[0.5, 0.1, 2, 1])
            for r in range(n_ex)]
    else:
        inner_grid = [
            outer_grid[r].subgridspec(2, 1, hspace=0, height_ratios=[2, 1]) for r in range(n_ex)]

    bout_ex_idxs = np.sort(np.random.permutation(bouts_clean.shape[0])[:n_ex])

    for ex in range(n_ex):

        col = ex % n_cols
        row = int(np.floor(ex / n_cols))
        axes = inner_grid[ex].subplots()

        idx_align = bouts_clean[bout_ex_idxs[ex], 1]
        slc = (idx_align - frame_win, idx_align + frame_win)

        a = -1

        # hand labels
        if states_hand is not None:
            a += 1
            states_aug = np.concatenate([
                states_hand[None, slice(*slc)], np.array([[0, n_colors - 1]])], axis=1)
            axes[a].imshow(
                states_aug, aspect='auto', interpolation='none', cmap=cmap,
                extent=(0, slc[1] - slc[0], 0, 1))
            axes[a].axvline(frame_win, color='k')
            if col == 0:
                axes[a].text(
                    -0.02, 0.25, 'hand', ha='right', transform=axes[a].transAxes, fontsize=10)
            axes[a].set_xticks([])
            axes[a].set_yticks([])

            # space
            a += 1
            axes[a].set_axis_off()

        # state predictions
        a += 1
        plot_markers_and_states(
            times=np.arange(slc[1] - slc[0]),
            markers=plotting_markers[slice(*slc)],
            states=states[slice(*slc)],
            marker_names=marker_names if col == 0 else None,
            cmap=cmap,
            title=None, ax=axes[a], ymin=ymin, ymax=ymax, spc=spc)
        axes[a].axvline(frame_win, color='k')
        axes[a].set_xticks([])

        # probabilities
        a += 1
        for l in range(probs.shape[1]):
            axes[a].plot(probs[slice(*slc), l], color=cmap.colors[l])
        axes[a].set_xlim([0, slc[1] - slc[0]])
        axes[a].axvline(frame_win, color='k')
        xticklabels_ = xticklabels + idx_align / framerate
        axes[a].set_xticks(xticks)
        axes[a].set_xticklabels(['%5.1f' % x for x in xticklabels_], fontsize=9)
        axes[a].tick_params(axis='x', which='major', pad=1)
        if col == 0:
            axes[a].set_yticks([0, 1])
            axes[a].set_yticklabels([0, 1], fontsize=10)
            axes[a].set_ylabel(
                'probability', rotation='horizontal', ha='right', va='center',
                fontsize=10)
        if row == (n_rows - 1):
            axes[a].set_xlabel('Time (sec)', fontsize=14)

    if n_rows < 3:
        top = 0.85
    elif n_rows < 2:
        top = 0.8
    else:
        top = 0.93

    fig.subplots_adjust(right=0.8)
    # lower left corner in [0.83, 0.85]
    # axes width 0.02 and height 0.10
    r = (7 * 0.25) / fig_height  # height of good bar / height of full examples sheet
    bar_height = r  # 0.10
    bar_bot = top - r  # 0.85
    cb_ax = fig.add_axes([0.83, bar_bot, 0.02, bar_height])
    sm = plt.cm.ScalarMappable(cmap=cmap)  # so that we don't have to use an axis output
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cb_ax)

    l = len(state_names)
    lp1 = 1 / (2 * l)
    cbar.set_ticks(np.linspace(lp1, 1 - lp1, l))
    cbar.ax.set_yticklabels(state_names)  # vertically oriented colorbar

    plt.subplots_adjust(top=top)
    if title is None:
        title = 'Behavioral bout onsets'
    plt.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file)

    # plt.show()
