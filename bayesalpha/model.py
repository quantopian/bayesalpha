from functools import partial

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
import random
import pandas as pd

from bayesalpha.dists import bspline_basis, GPExponential, NormalNonZero
from bayesalpha.serialize import to_xarray
from bayesalpha._version import get_versions
from bayesalpha.plotting import plot_horizontal_dots

_PARAM_DEFAULTS = {
    'shrinkage': 'exponential'
}


def build_model(data, algos, **params):
    data = data.fillna(0.)
    params = params.copy()

    is_author_is = np.zeros(data.shape, dtype=np.int8)
    for i, algo in enumerate(data):
        if algo not in algos.index or pd.isnull(algos.created_at.loc[algo]):
            raise ValueError('No `created_at` value for algo %s' % algo)
        is_author_is[:, i] = data.index < algos.created_at.loc[algo]

    duration = data.index[-1] - data.index[0]

    n, k = data.shape

    # Find knot positions for gains and vlt splines
    n_knots_vlt = duration.days // 5
    time_vlt = np.linspace(0, 1, n)
    Bx_log_vlt = bspline_basis(n_knots_vlt, time_vlt)
    Bx_log_vlt = sparse.csr_matrix(Bx_log_vlt)

    n_knots_gains = duration.days // 10
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

        shrinkage = params.pop('shrinkage')

        # Define shrinkage model on the long-term gains
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
            gains_raw = pm.Normal('gains_raw', shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'trace-exponential':
            mu = params.pop('gains_sd_trace_mu')
            sd = params.pop('gains_sd_trace_sd')
            gains_sd = pm.Bound(pm.Normal, lower=0)('gains_sd', mu=mu, sd=sd)
            gains_raw = pm.Laplace('gains_raw', mu=0, b=1, shape=k)
            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'trace-normal':
            mu = params.pop('gains_sd_trace_mu')
            sd = params.pop('gains_sd_trace_sd')
            gains_sd = pm.Normal('gains_sd', mu=mu, sd=sd)
            gains_raw = pm.Normal('gains_raw', shape=k)
            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        else:
            raise ValueError('Unknown gains model: %s' % shrinkage)

        # Define variations of gains over time
        gains_time_alpha = pm.HalfNormal('gains_time_alpha', sd=0.1)
        if 'gains_time_sd_sd_trace_mu' in params:
            mu = params.pop('gains_time_sd_sd_trace_mu')
            sd = params.pop('gains_time_sd_sd_trace_sd')
            BoundNormal = pm.Bound(pm.Normal, lower=0)
            gains_time_sd_sd = BoundNormal('gains_time_sd_sd', mu=mu, sd=sd)
        else:
            gains_time_sd_sd = pm.HalfStudentT('gains_time_sd_sd', nu=3, sd=0.1)
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
    def timestamp(self):
        return self._trace.attrs['timestamp']

    @property
    def model_version(self):
        return self._trace.attrs['model-version']

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

    @property
    def seed(self):
        return self._trace.attrs['seed']

    @property
    def id(self):
        hasher = hashlib.sha256()
        hasher.update(self.params_hash.encode())
        hasher.update(self.model_version.encode())
        hasher.update(str(self.seed).encode())
        return hasher.hexdigest()[:16]

    def raise_ok(self):
        if not self.ok:
            warnings = self.warnings
            raise RuntimeError('Problems during sampling: %s' % warnings)

    def plot_prob(self, algos=None, ax=None, sort=True, rope=False, rope_upper=.05, rope_lower=None):
        if rope:
            prob_func = partial(self.gains_rope, rope_upper, lower=rope_lower)
        else:
            prob_func = self.gains_pos_prob

        if algos is not None:
            vals = prob_func().loc[algos]
        else:
            vals = prob_func()

        xlabel = 'P(gains ~ 0)' if rope else 'P(gains > 0)'

        ax = plot_horizontal_dots(vals, sort=sort, ax=ax, xlabel=xlabel)

        return ax

    def gains_pos_prob(self):
        return (
            (self.trace['gains'] > 0)
            .mean(['sample', 'chain'])
            .to_series()
            .rename('gains_pos_prob'))

    def gains_rope(self, upper, lower=None):
        if lower is None:
            lower = -upper
        return (
            ((self.trace['gains'] > lower) & (self.trace['gains'] < upper))
            .mean(['sample', 'chain'])
            .to_series()
            .rename('gains_rope'))


def fit_population(data, algos, sampler_args=None, save_data=True,
                   seed=None, **params):
    """Fit the model to daily returns.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the returns. Columns are the different
        algos, rows the days. If an algorithm doesn't do anything on a day,
        that value can be NaN.
    algos : pd.DataFrame
        Dataframe containing metadata about the algorithms. It must contain
        a column 'created_at', with the dates when the algorithm was created.
        All later daily returns are interpreted as author-out-of-sample.
    sampler_args : dict
        Additional parameters for `pm.sample`.
    save_data : bool
        Whether to store the dataset in the result object.
    seed : int
        Seed for random number generation in PyMC3.
    """
    _check_data(data)
    params_ = _PARAM_DEFAULTS.copy()
    params_.update(params)
    params = params_

    if sampler_args is None:
        sampler_args = {}
    else:
        sampler_args = sampler_args.copy()
    if seed is None:
        seed = random.getrandbits(32)
    if 'random_seed' in sampler_args:
        raise ValueError('Can not specify `random_seed`.')
    sampler_args['random_seed'] = seed

    model, coords, dims = build_model(data, algos, **params)
    timestamp = datetime.isoformat(datetime.now())
    with model:
        args = {} if sampler_args is None else sampler_args
        with warnings.catch_warnings(record=True) as warns:
            trace = pm.sample(**args)
    if warns:
        warnings.warn('Problems during sampling. Inspect `result.warnings`.')
    trace = to_xarray(trace, coords, dims)
    trace.attrs['params'] = json.dumps(params)
    trace.attrs['timestamp'] = timestamp
    trace.attrs['warnings'] = json.dumps([str(warn) for warn in warns])
    trace.attrs['seed'] = seed
    trace.attrs['model-version'] = get_versions()['version']

    if save_data:
        trace.coords['algodata'] = algos.columns
        trace['_data'] = (('time', 'algo'), data)
        try:
            trace['_algos'] = (('algo', 'algodata'), algos.loc[data.columns])
        except ValueError:
            warnings.warn('Could not save algo metadata, skipping.')
    return FitResult(trace)


_TRACE_PARAM_NAMES = {
    'normal': (
        'trace-normal',
        ['gains_sd', 'gains_time_sd_sd']),
    'exponential': (
        'trace-exponential',
        ['gains_sd', 'gains_time_sd_sd']
    )
}


def fit_single(data, algos, population_fit=None, sampler_args=None, seed=None,
               **params):
    """Fit the model to algorithms and use an earlier run for hyperparameters.

    Use a model fit with a large number of algorithms to get estimates
    for global parameters -- for example those that inform about the
    distribution of gain parameters.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the returns. Columns are the different
        algos, rows the days. If an algorithm doesn't do anything on a day,
        that value can be NaN.
    algos : pd.DataFrame
        Dataframe containing metadata about the algorithms. It must contain
        a column 'created_at', with the dates when the algorithm was created.
        All later daily returns are interpreted as author-out-of-sample.
    population_fit : FitResult
        The result of a previous model fit using `fit_population`. If this
        is not specified, all necessary parameters have to be specified as
        keyword arguments.
    sampler_args : dict
        Additional arguments for `pm.sample`
    seed : int
        Seed for random numbers during sampling.
    """
    params = params.copy()

    if population_fit is None:
        trace_shrinkage = None
    else:
        trace_shrinkage = population_fit.params['shrinkage']

    shrinkage = params.pop('shrinkage', trace_shrinkage)
    if shrinkage is None:
        raise ValueError('Either `shrinkage` or `population_fit` has to be '
                         'specified.')
    if trace_shrinkage is not None and shrinkage != trace_shrinkage:
        raise ValueError('Can not use different shrinkage type in population '
                         'and single algo fit.')

    if shrinkage not in _TRACE_PARAM_NAMES:
        raise ValueError('Can not fit single algo for shrinkage %s' % shrinkage)
    shrinkage, param_names = _TRACE_PARAM_NAMES[shrinkage]

    for name in param_names:
        name_mu = name + '_trace_mu'
        name_sd = name + '_trace_sd'
        if name_mu in params and name_sd in params:
            continue
        if population_fit is None:
            raise ValueError('population_fit or %s and %s must be specified.'
                             % (name_mu, name_sd))
        trace_vals = population_fit.trace[name]
        params.setdefault(name_mu, float(trace_vals.mean()))
        params.setdefault(name_sd, float(trace_vals.std()))

    fit = fit_population(data, algos=algos, sampler_args=sampler_args,
                         seed=seed, shrinkage=shrinkage, **params)
    if population_fit is not None:
        parent = population_fit.trace
        fit.trace.attrs['parent-params'] = parent.attrs['params']
        fit.trace.attrs['parent-seed'] = parent.attrs['seed']
        fit.trace.attrs['parent-version'] = parent.attrs['model-version']
        fit.trace.attrs['parent-id'] = population_fit.id
    return fit


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
