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


