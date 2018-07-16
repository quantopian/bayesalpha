import xarray as xr
import numpy as np


def to_xarray(trace, coords, dims):
    """Convert a pymc3 trace to an xarray dataset.

    Parameters
    ----------
    trace : pymc3 trace
    coords : dict
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, Tuple(str)]
        A mapping from pymc3 variables to a tuple corresponding to
        the shape of the variable, where the elements of the tuples are
        the names of the coordinate dimensions.

    Example
    -------
    ::

        coords = {
            'subject': ['Peter', 'Hans'],
            'time': [Timestamp('2017-01-20'), Timestamp('2017-01-21')],
            'treatment': ['sorafenib', 'whatever']
        }
        dims = {
            'subject_mu': ('subject',),
            'effect': ('treatment',),
            'interaction': ('time', 'treatment'),
        }
    """
    coords = coords.copy()
    coords['sample'] = list(range(len(trace)))
    coords['chain'] = list(range(trace.nchains))

    coords_ = {}
    for key, vals in coords.items():
        coords_[key] = xr.IndexVariable((key,), data=vals)
    coords = coords_

    data = xr.Dataset(coords=coords)
    for key in trace.varnames:
        if key.endswith('_'):
            continue
        dims_str = ('chain', 'sample')
        if key in dims:
            dims_str = dims_str + dims[key]
        vals = trace.get_values(key, combine=False, squeeze=False)
        vals = np.array(vals)
        data[key] = xr.DataArray(vals,
                                 {v: coords[v] for v in dims_str},
                                 dims=dims_str)

    return data


def xarray_hash():
    pass
