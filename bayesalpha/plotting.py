import matplotlib.pyplot as plt
import numpy as np

def plot_horizontal_dots(vals, sort=True, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 7))

    if sort:
        vals = vals.sort_values()

    y = -np.arange(len(vals))

    plot_kwargs = dict(xlim=(0, 1),
                       xlabel=xlabel,
                       yticks=y,
                       yticklabels=vals.index,
                       ylim=(-len(y) + .5, .5))

    plot_kwargs.update(kwargs)

    ax.grid(axis='x', color='w', zorder=-5)
    ax.scatter(vals.values, y, marker='d', zorder=5)
    ax.axvline(0.5, alpha=0.3, color='black')

    locs = y

    ax.set(**plot_kwargs)

    ax.barh(locs, [max(ax.get_xticks())] * len(locs),
            height=(locs[1]-locs[0]),
            color=['lightgray', 'w'],
            zorder=-10, alpha=.25)


    return ax
