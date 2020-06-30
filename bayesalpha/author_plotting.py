import numpy as np
import warnings
import functools
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _has_mpl = True
except ImportError:
    warnings.warn('Could not import matplotlib: Plotting unavailable.')
    _has_mpl = False
    plt = None
    sns = None


def _require_mpl(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        if not _has_mpl:
            raise RuntimeError('Matplotlib is unavailable.')
        return func(*args, **kwargs)

    return inner


@_require_mpl
def plot_trace(trace, varname, title=None, ax=None, **kwargs):
    """
    Plot samples from trace for a specific variable.

    Parameters
    ----------
    trace : AuthorModelResult object
        Result from ba.fit_authors
    varname : str
        Name of variable to plot. Must be one of ['mu_global', 'mu_author',
        'mu_algo', 'alpha_author', 'alpha_algo']
    title : str (optional)
        Title of plot
    ax : plt.axis object (optional)
        Axis on which to plot
    kwargs : dict (optional)
        Additional keyword args to pass to sns.distplot
    """

    if varname not in ['mu_global', 'mu_author', 'mu_algo',
                       'alpha_author', 'alpha_algo']:
        raise ValueError("`varname` must be one of ['mu_global', 'mu_author', "
                         "'mu_algo', 'alpha_author', 'alpha_algo']")

    if ax is None:
        _, ax = plt.subplots(figsize=[12, 4])

    for i in trace.trace[varname]['chain']:
        if varname == 'mu_global':
            sns.distplot(trace.trace['mu_global'].sel({'chain': i}).values,
                         **kwargs)
        else:
            suffix = varname.split('_')[-1]  # Either 'author' or 'algo'
            for j in trace.trace[varname][suffix]:
                sns.distplot(trace.trace[varname].sel({'chain': i,
                                                       suffix: j}).values,
                             **kwargs)

    if title:
        ax.set_title(title)

    plt.xlabel(varname)
    plt.ylabel('Probability')

    return ax
