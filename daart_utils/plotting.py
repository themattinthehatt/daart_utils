"""daart plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_heatmaps(
        df, metric, expt_ids, title='', kind=1, n_cols=2, center=False, vmin=0, vmax=1, agg='mean',
        annot=False, cmaps=None, save_file=None, **kwargs):
    """Plot a heatmap of model performance as a function of hyperparameters

    Parameters
    ----------
    df : pd.DataFrame
        must include the following columns: label, lambda_weak, lambda_pred, `metric`, expt_id
    metric : str
        metric whose values will be displayed with heatmap; column name in `df`
    expt_ids : list
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
        for expt_id in expt_ids:

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 1, 4 * n_rows + 1))
            axes = axes.flatten()
            for ax in axes:
                ax.set_axis_off()
            for l, label_name in enumerate(label_names):
                axes[l].set_axis_on()
                data_tmp = df[(df.label == label_name) & (df.expt_id == expt_id)].pivot(
                    'lambda_weak', 'lambda_pred', metric)
                sns.heatmap(data_tmp, vmin=vmin, vmax=vmax, ax=axes[l], annot=annot, **kwargs)
                axes[l].invert_yaxis()
                axes[l].set_title(label_name)
            fig.suptitle('%s %s' % (expt_id, title))
            plt.tight_layout()
            plt.show()

    elif kind == 2:

        # plot heatmap for each label, averaged over expt ids
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 1, 4 * n_rows + 1))
        axes = axes.flatten()
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
