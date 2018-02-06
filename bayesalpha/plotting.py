import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_height(k):
    return 2 + 14 * (1 - np.exp(-0.02 * k))


def plot_horizontal_dots(vals, sort=True, ax=None, title=None, **kwargs):
    if ax is None:
        height = get_height(len(vals))
        _, ax = plt.subplots(1, 1, figsize=(4, height))

    if sort:
        vals = vals.sort_values()

    y = -np.arange(len(vals))

    plot_kwargs = dict(xlim=(0, 1),
                       yticks=y,
                       yticklabels=vals.index,
                       ylim=(-len(y) + .5, .5))

    plot_kwargs.update(kwargs)

    ax.grid(axis='x', color='w', zorder=-5)
    ax.scatter(vals.values, y, marker='d', zorder=5)
    ax.axvline(0.5, alpha=0.3, color='black')
    if title:
        ax.set_title(title)

    locs = y

    ax.set(**plot_kwargs)

    if len(vals) > 1:
        ax.barh(locs, [max(ax.get_xticks())] * len(locs),
                height=(locs[1] - locs[0]),
                color=['lightgray', 'w'],
                zorder=-10, alpha=.25)

    return ax


def plot_correlations(corr_xarray, corr_threshold=.33, ax=None, cmap=None, **heatmap_kwargs):
    k = len(corr_xarray.coords['algo'])
    if ax is None:
        w, h = get_height(k)*2+1, get_height(k)*2
        fig, ax = plt.subplots(2, 2, figsize=(w, h))
    else:
        fig = None
    _cmap = dict(
        mean='bwr',
        std='magma',
        prob='pink',
        snr='PuOr'
    )
    if cmap is not None:
        _cmap.update(cmap)
    ax = ax.flat
    mean = corr_xarray.mean(['chain', 'sample']).values
    std = corr_xarray.std(['chain', 'sample']).values
    prob = (abs(corr_xarray) > corr_threshold).mean(['chain', 'sample']).values
    snr = mean / std
    snr[range(k), range(k)] = 0
    mask = np.zeros(mean.shape, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    kwargs = dict(mask=mask, square=True, linewidths=.5)
    kwargs.update(heatmap_kwargs)
    sns.heatmap(mean, ax=ax[0], vmin=-1, vmax=1, cmap=_cmap['mean'], **kwargs)
    ax[0].set_title('E[corr]')
    sns.heatmap(std, ax=ax[1], cmap=_cmap['std'], **kwargs)
    ax[1].set_title('std[corr]')
    sns.heatmap(prob, ax=ax[2], vmin=0, vmax=1, cmap=_cmap['prob'], **kwargs)
    ax[2].set_title('P(|corr|>{})'.format(corr_threshold))
    sns.heatmap(snr, ax=ax[3], center=0, cmap=_cmap['snr'], **kwargs)
    ax[3].set_title('E[corr]/std[corr]')
    for i in range(len(ax)):
        ax[i].set_xticklabels(corr_xarray.coords['algo'].values, rotation=90)
        ax[i].set_yticklabels(corr_xarray.coords['algo'].values, rotation=0)
    if fig is not None:
        fig.tight_layout()
    return ax.base
