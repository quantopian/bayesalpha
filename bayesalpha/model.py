import numpy as np
from scipy import sparse
import theano
import theano.tensor as tt
import pymc3 as pm
import json
import hashlib
import xarray as xr
from datetime import datetime
import warnings
import matplotlib.pyplot as plt

from bayesalpha.dists import bspline_basis, GPExponential, NormalNonZero
from bayesalpha.serialize import xarray_hash, to_xarray


_PARAM_DEFAULTS = {
    'shrinkage': 'exponential'
}


def build_model(data, algos=None, is_author_is=None, *, shrinkage, **params):
    data = data.fillna(0.)
    params = params.copy()

    if is_author_is is None:
        is_author_is = np.zeros(data.shape, dtype=np.int8)
        for i, algo in enumerate(data):
            if algo not in algos.index:
                is_author_is[:len(is_author_is) // 2, i] = True
            else:
                is_author_is[:, i] = data.index < algos.created_at.loc[algo]

    duration = data.index[-1] - data.index[0]

    n, k = data.shape

    # Find knot positions for gains and vlt splines
    n_knots_vlt = duration.days // 5
    time_vlt = np.linspace(0, 1, n)
    Bx_log_vlt = bspline_basis(n_knots_vlt, time_vlt)
    Bx_log_vlt = sparse.csr_matrix(Bx_log_vlt)

    n_knots_gains = duration.days // 20
    time_gains = np.linspace(0, 1, n)
    Bx_gains = bspline_basis(n_knots_gains, time_gains)
    Bx_gains = sparse.csr_matrix(Bx_gains)

    with pm.Model() as model:
        Bx_log_vlt_ = theano.sparse.as_sparse_variable(Bx_log_vlt)
        Bx_gains_ = theano.sparse.as_sparse_variable(Bx_gains)

        # Model the log volatility as gauss process with splines interpolation
        log_vlt_time_alpha = pm.HalfNormal('log_vlt_time_alpha', sd=0.1)
        log_vlt_time_sd = pm.HalfNormal('log_vlt_time_sd', sd=0.5, shape=k)
        log_vlt_mu = pm.Normal('log_vlt_mu', mu=-3, sd=1, shape=k)
        log_vlt_time_raw = GPExponential(
            'log_vlt_time_raw', mu=0, alpha=log_vlt_time_alpha,
            sigma=1, shape=(k, n_knots_vlt))
        log_vlt_raw = log_vlt_mu[:, None] + log_vlt_time_sd[:, None] * log_vlt_time_raw
        log_vlt = theano.sparse.dot(Bx_log_vlt_, log_vlt_raw.T).T
        log_vlt = pm.Deterministic('log_vlt', log_vlt)
        vlt = tt.exp(log_vlt)

        # Define shrinkage model on the long-time gains
        if shrinkage == 'exponential-mix':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.2)
            gains_theta = pm.Exponential('gains_theta', lam=1, shape=k)
            gains_eta = pm.Normal('gains_eta', shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = gains_sd * gains_theta * gains_eta
            gains = pm.Deterministic('gains', gains)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'exponential':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.1)
            gains_raw = pm.Laplace('gains_raw', mu=0, b=1, shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'student':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.2)
            gains_raw = pm.StudentT('gains_raw', nu=4, mu=0, sd=1, shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'normal':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.2)
            gains_raw = pm.Normal('gains_raw', mu=0, sd=1, shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'trace-normal':
            if 'gains_mu' not in params or 'gains_sd' not in params:
                raise ValueError('gains_mu and gains_sd must be '
                                 'specified for trace-normal shrinkage.')
            gains_mu = params.pop('gains_mu')
            gains_sd = params.pop('gains_sd')
            gains_raw = pm.Normal('gains_raw', mu=0, sd=1, shape=k)
            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic(
                'gains', gains_sd * gains_raw + gains_mu)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        else:
            raise ValueError('Unknown gains model: %s' % shrinkage)

        # Define variations of gains over time
        gains_time_alpha = pm.HalfNormal('gains_time_alpha', sd=0.1)
        gains_time_sd_sd = params.pop('gains_time_sd_sd', None)
        if gains_time_sd_sd is None:
            gains_time_sd_sd = pm.HalfStudentT('gains_time_sd_sd', nu=3, sd=0.1)
        else:
            gains_time_sd_sd = tt.as_tensor_variable(gains_time_sd_sd)
            gains_time_sd_sd = pm.Deterministic('gains_time_sd_sd', gains_time_sd_sd)
        gains_time_sd_raw = pm.HalfStudentT('gains_time_sd_raw', nu=3, sd=1, shape=k)
        gains_time_sd = pm.Deterministic(
            'gains_time_sd', gains_time_sd_sd * gains_time_sd_raw)
        gains_time_raw = GPExponential(
            'gains_time_raw', mu=0, alpha=gains_time_alpha,
            sigma=1, shape=(k, n_knots_gains))
        gains_time = gains_time_sd[:, None] * gains_time_raw
        gains_time = theano.sparse.dot(Bx_gains_, gains_time.T).T
        gains_time = pm.Deterministic('gains_time', gains_time)

        mu = (gains_all.T + gains_time) * vlt

        observed = data.values.T
        NormalNonZero('y', mu=mu, sd=vlt, observed=observed)

        if params:
            raise ValueError('Unused params: %s' % params.keys())

        coords = {
            'algo': data.columns,
            'time': data.index,
            'time_raw_gains': list(range(n_knots_gains)),
            'time_raw_vlt': list(range(n_knots_vlt)),
        }

        dims = {
            'log_vlt_time_sd': ('algo',),
            'log_vlt_mu': ('algo',),
            'log_vlt_time_raw': ('algo', 'time_raw_vlt'),
            'log_vlt': ('algo', 'time'),
            'gains_theta': ('algo',),
            'gains_eta': ('algo',),
            'author_is': ('algo',),
            'gains': ('algo',),
            'gains_raw': ('algo',),
            'gains_time_sd_raw': ('algo',),
            'gains_time_sd': ('algo',),
            'gains_time_raw': ('algo', 'time_raw_gains'),
            'gains_time': ('algo', 'time'),
        }

        return model, coords, dims


class FitResult:
    def __init__(self, trace):
        self._trace = trace

    def save(self, filename, group, **args):
        """Save the results to a netcdf file."""
        self._trace.to_netcdf(filename, group=group, **args)

    @classmethod
    def _load(cls, filename, group=None):
        trace = xr.open_dataset(filename, group=group)
        return cls(trace=trace)

    @property
    def trace(self):
        return self._trace

    @property
    def params(self):
        return json.loads(self._trace.attrs['params'])

    @property
    def result_hash(self):
        return xarray_hash(self._trace)

    @property
    def timestamp(self):
        return self._trace.attrs['timestamp']

    @property
    def params_hash(self):
        params = json.dumps(self.params, sort_keys=True)
        hasher = hashlib.sha256(params.encode())
        return hasher.hexdigest()[:16]

    @property
    def ok(self):
        return len(self.warnings) == 0

    @property
    def warnings(self):
        return json.loads(self._trace.attrs['warnings'])

    def raise_ok(self):
        if not self.ok:
            warnings = self.warnings
            raise RuntimeError('Problems during sampling: %s' % warnings)

    def plot_gains_pos_prob(self, algos=None, ax=None, sort=True):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(4, 7))
        if algos is not None:
            vals = self.gains_pos_prob().loc(algo=algos)
        else:
            vals = self.gains_pos_prob()

        if sort:
            vals = vals.sort_values()

        y = -np.arange(len(vals))
        ax.grid(axis='x', color='w', zorder=-5)
        ax.scatter(vals.values, y, marker='d', zorder=5)
        ax.axvline(0.5, alpha=0.3, color='black')
        ax.set_xlim(0, 1)
        locs = y
        ax.barh(locs, [max(ax.get_xticks())] * len(locs),
                height=(locs[1]-locs[0]),
                color=['lightgray', 'w'],
                zorder=-10, alpha=.25)
        ax.set_yticks(y)
        ax.set_yticklabels(vals.index)
        ax.set_ylim(-len(y) + .5, .5)
        return ax

    def gains_pos_prob(self):
        return (
            (self.trace['gains'] > 0)
            .mean(['sample', 'chain'])
            .to_series()
            .rename('gains_pos_prob'))

    def gains_rope(self, upper):
        return (
            (self.trace['gains'] > upper)
            .mean(['sample', 'chain'])
            .to_series()
            .rename('gains_rope'))


def fit_population(data, *, algos=None, is_author_is=None, sampler_args=None, **params):
    """Fit the model to daily returns.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the returns. Columns are the different
        algos, rows the days. If an algorithm doesn't do anything on a day,
        that value can be NaN.
    algos : pd.DataFrame
        Dataframe containing metadata about the algorithms. If specified,
        it must contain a column 'created_at', with the dates when
        the algorithm was created. All later daily returns are interpreted
        as author-out-of-sample.
    is_author_is : np.array
        Alternatively to `algos`, the author-out-of-sample times can be
        specified as a boolean array with shape (n_time, n_algos).
    sample_args : dict
        Additional parameters for `pm.sample`.
    """
    if (algos is None) and (is_author_is is None):
        raise ValueError('Exactly one of `algos` and `is_author_is` must be '
                         'specified.')
    _check_data(data)
    params_ = _PARAM_DEFAULTS.copy()
    params_.update(params)
    params = params_
    model, coords, dims = build_model(data, algos, is_author_is, **params)
    timestamp = datetime.isoformat(datetime.now())
    with model:
        args = {} if sampler_args is None else sampler_args
        with warnings.catch_warnings(record=True) as warns:
            trace = pm.sample(**args)
    trace = to_xarray(trace, coords, dims)
    trace.attrs['params'] = json.dumps(params)
    trace.attrs['timestamp'] = timestamp
    trace.attrs['warnings'] = json.dumps([str(warn) for warn in warns])
    return FitResult(trace)


def fit_single(data, *, population_fit=None, algos=None, is_author_is=None,
               sampler_args=None, **params):
    params = params.copy()
    shrinkage = params.setdefault('shrinkage', 'trace-normal')
    if shrinkage != 'trace-normal':
        raise ValueError('shrinkage can not be specified for single algo run.')

    # TODO some kind of normal test?
    if any(var not in params
           for var in ['gains_mu', 'gains_sd', 'gains_time_sd_sd']):
        if population_fit is None:
            raise ValueError('population_fit or all of `gains_mu`, `gains_sd` '
                             'and `gains_time_sd_sd` must be specified.')
        population = population_fit.trace
        popgains = population['gains']
        pop_mutime = population['gains_time_sd_sd']
        params.setdefault('gains_mu', float(popgains.mean()))
        params.setdefault('gains_sd', float(popgains.std()))
        params.setdefault('gains_time_sd_sd', float(pop_mutime.mean()))
    return fit_population(data, algos=algos, is_author_is=is_author_is,
                          sampler_args=sampler_args, **params)


def load(filename, group):
    """Load results from an netcdf file."""
    return FitResult._load(filename, group)


def _check_data(data):
    if data.count().min() < 100:
        warnings.warn('The dataset contains algos with fewer than 100 '
                      'observations.')
    if not data.index.dtype_str.startswith('datetime'):
        raise ValueError('Index of dataset must have a datetime dtype')
    if (np.abs(data) > 0.2).any().any():
        raise ValueError('Dataset contains unrealistically large returns.')
    if (~np.isfinite(data.fillna(0.))).any().any():
        raise ValueError('Dataset contains inf.')
