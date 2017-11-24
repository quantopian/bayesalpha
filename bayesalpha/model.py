from functools import partial
import warnings

import numpy as np
from scipy import sparse, stats
import theano
import theano.tensor as tt
import pymc3 as pm
import json
import hashlib
import xarray as xr
from datetime import datetime
import random
import pandas as pd

from bayesalpha.dists import bspline_basis, GPExponential, NormalNonZero
from bayesalpha.dists import dot as sparse_dot
from bayesalpha.serialize import to_xarray
from bayesalpha._version import get_versions
from bayesalpha.plotting import plot_horizontal_dots

_PARAM_DEFAULTS = {
    'shrinkage': 'exponential'
}


class ModelBuilder(object):
    def __init__(self, data, algos, **params):
        self.data = data
        # The build functions pop parameters they use
        self.params = params.copy()
        self.algos = algos
        # The build functions add items when appropriate
        self.coords = {
            'algo': data.columns,
            'time': data.index,
        }
        # The build functions add items when appropriate
        self.dims = {}
        self.n_algos = len(data.columns)
        self.n_time = len(data.index)

        self.model = pm.Model()
        with self.model:
            in_sample = self._build_in_sample()
            Bx_log_vlt, Bx_gains = self._build_splines()

            vlt = self._build_volatility(Bx_log_vlt)

            gains_mu = self._build_gains_mu(in_sample)
            gains_time = self._build_gains_time(Bx_gains)

            mu = (gains_mu + gains_time) * vlt

            self._build_likelihood(mu, vlt, observed=data.values.T)

        if self.params:
            raise ValueError('Unused params: %s' % params.keys())

    def _build_in_sample(self):
        data, algos = self.data, self.algos
        is_author_is = np.zeros(data.shape, dtype=np.int8)
        for i, algo in enumerate(data):
            isnull = pd.isnull(algos.created_at.loc[algo])
            if algo not in algos.index or isnull:
                raise ValueError('No `created_at` value for algo %s' % algo)
            is_author_is[:, i] = data.index < algos.created_at.loc[algo]
        return is_author_is

    def _build_splines(self):
        data = self.data
        n, k = data.shape

        duration = data.index[-1] - data.index[0]

        # Find knot positions for gains and vlt splines
        n_knots_vlt = duration.days // 5
        time_vlt = np.linspace(0, 1, n)
        Bx_log_vlt = bspline_basis(n_knots_vlt, time_vlt)
        Bx_log_vlt = sparse.csr_matrix(Bx_log_vlt)

        n_knots_gains = duration.days // 10
        time_gains = np.linspace(0, 1, n)
        Bx_gains = bspline_basis(n_knots_gains, time_gains)
        Bx_gains = sparse.csr_matrix(Bx_gains)

        Bx_log_vlt = theano.sparse.as_sparse_variable(Bx_log_vlt)
        Bx_gains = theano.sparse.as_sparse_variable(Bx_gains)

        self.coords['time_raw_gains'] = list(range(n_knots_gains))
        self.coords['time_raw_vlt'] = list(range(n_knots_vlt))

        return Bx_log_vlt, Bx_gains

    def _build_volatility(self, Bx_log_vlt):
        k = self.n_algos
        n_knots_vlt = len(self.coords['time_raw_vlt'])

        log_vlt_time_alpha = pm.HalfNormal('log_vlt_time_alpha', sd=0.1)
        log_vlt_time_sd = pm.HalfNormal('log_vlt_time_sd', sd=0.5, shape=k)
        self.dims['log_vlt_time_sd'] = ('algo',)

        log_vlt_mu = pm.Normal('log_vlt_mu', mu=-3, sd=1, shape=k)
        self.dims['log_vlt_mu'] = ('algo',)
        log_vlt_time_raw = GPExponential(
            'log_vlt_time_raw', mu=0, alpha=log_vlt_time_alpha,
            sigma=1, shape=(k, n_knots_vlt))
        self.dims['log_vlt_time_raw'] = ('algo', 'time_raw_vlt')
        log_vlt_raw = (log_vlt_mu[:, None]
                       + log_vlt_time_sd[:, None] * log_vlt_time_raw)
        log_vlt = sparse_dot(Bx_log_vlt, log_vlt_raw.T)

        pm.Deterministic('log_vlt', log_vlt)
        self.dims['log_vlt'] = ('algo', 'time')
        vlt = tt.exp(log_vlt)
        return vlt

    def _build_gains_mu(self, is_author_is):
        self.dims.update({
            'gains_theta': ('algo',),
            'gains_eta': ('algo',),
            'author_is': ('algo',),
            'gains': ('algo',),
            'gains_raw': ('algo',),
        })
        shrinkage = self.params.pop('shrinkage')
        k = self.n_algos

        # Define shrinkage model on the long-term gains
        if shrinkage == 'exponential-mix':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.2)
            pm.Deterministic('log_gains_sd', tt.log(gains_sd))
            gains_theta = pm.Exponential('gains_theta', lam=1, shape=k)
            gains_eta = pm.Normal('gains_eta', shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = gains_sd * gains_theta * gains_eta
            gains = pm.Deterministic('gains', gains)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'exponential':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.1)
            pm.Deterministic('log_gains_sd', tt.log(gains_sd))
            gains_raw = pm.Laplace('gains_raw', mu=0, b=1, shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'student':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.2)
            pm.Deterministic('log_gains_sd', tt.log(gains_sd))
            gains_raw = pm.StudentT('gains_raw', nu=4, mu=0, sd=1, shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'normal':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.2)
            pm.Deterministic('log_gains_sd', tt.log(gains_sd))
            gains_raw = pm.Normal('gains_raw', shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'trace-exponential':
            mu = self.params.pop('log_gains_sd_trace_mu')
            sd = self.params.pop('log_gains_sd_trace_sd')
            log_gains_sd = pm.Normal('log_gains_sd', mu=mu, sd=sd)
            gains_sd = pm.Deterministic('gains_sd', tt.exp(log_gains_sd))
            gains_raw = pm.Laplace('gains_raw', mu=0, b=1, shape=k)
            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'trace-normal':
            mu = self.params.pop('log_gains_sd_trace_mu')
            sd = self.params.pop('log_gains_sd_trace_sd')
            log_gains_sd = pm.Normal('log_gains_sd', mu=mu, sd=sd)
            gains_sd = pm.Deterministic('gains_sd', tt.exp(log_gains_sd))
            gains_raw = pm.Normal('gains_raw', shape=k)
            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        else:
            raise ValueError('Unknown gains model: %s' % shrinkage)

        return gains_all.T

    def _build_gains_time(self, Bx_gains):
        self.dims.update({
            'gains_time_sd_raw': ('algo',),
            'gains_time_sd': ('algo',),
            'log_gains_time_sd': ('algo',),
            'gains_time_raw': ('algo', 'time_raw_gains'),
            'gains_time': ('algo', 'time'),
        })
        k = self.n_algos
        n_knots_gains = len(self.coords['time_raw_gains'])

        gains_time_alpha = pm.HalfNormal('gains_time_alpha', sd=0.1)
        if 'log_gains_time_sd_sd_trace_mu' in self.params:
            mu = self.params.pop('log_gains_time_sd_sd_trace_mu')
            sd = self.params.pop('log_gains_time_sd_sd_trace_sd')
            log_gains_time_sd_sd = pm.Normal(
                'log_gains_time_sd_sd', mu=mu, sd=sd)
            gains_time_sd_sd = pm.Deterministic(
                'gains_time_sd_sd', tt.exp(log_gains_time_sd_sd))
        else:
            gains_time_sd_sd = pm.HalfStudentT(
                'gains_time_sd_sd', nu=3, sd=0.1)
            pm.Deterministic('log_gains_time_sd_sd', tt.log(gains_time_sd_sd))
        gains_time_sd_raw = pm.HalfStudentT(
            'gains_time_sd_raw', nu=3, sd=1, shape=k)
        gains_time_sd = pm.Deterministic(
            'gains_time_sd', gains_time_sd_sd * gains_time_sd_raw)
        gains_time_raw = GPExponential(
            'gains_time_raw', mu=0, alpha=gains_time_alpha,
            sigma=1, shape=(k, n_knots_gains))
        gains_time = gains_time_sd[:, None] * gains_time_raw
        gains_time = sparse_dot(Bx_gains, gains_time.T)

        pm.Deterministic('gains_time', gains_time)
        return gains_time

    def _build_likelihood(self, mu, sd, observed):
        NormalNonZero('y', mu=mu, sd=sd, observed=observed)
        if self.params.pop('save_mu', False):
            self.dims['mu'] = ('algo', 'time')
            pm.Deterministic('mu', mu)
        if self.params.pop('save_vlt', False):
            self.dims['vlt'] = ('algo', 'time')
            pm.Deterministic('vlt', sd)


def build_model(data, algos, **params):
    builder = ModelBuilder(data, algos, **params)
    return builder.model, builder.coords, builder.dims


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

    def plot_prob(self, algos=None, ax=None, sort=True, rope=False,
                  rope_upper=.05, rope_lower=None):
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

    def _points(self):
        for chain in self.trace.chain:
            for sample in self.trace.sample:
                data = {}
                for var in self.trace.data_vars:
                    if var.startswith('_'):
                        continue
                    vals = self.trace[var].sel(chain=chain, sample=sample)
                    data[var] = vals.values
                yield (chain, sample), data

    def predict(self, n_days):
        warnings.warn('The interface of predict will change...')
        start = pd.Timestamp(self.trace.time[0].values)
        start += pd.tseries.offsets.BDay(1)
        end_ = start + pd.tseries.offsets.BDay(2 * n_days)
        index = pd.date_range(start, end_, freq='B')[:n_days]
        columns = self.trace.algo

        data = pd.DataFrame(index=index, columns=columns)
        data.values[...] = 0.
        algos = self.trace._algos.to_pandas().copy()
        algos['created_at'] = start
        params = self.params.copy()
        params['save_mu'] = True
        params['save_vlt'] = True
        model = ModelBuilder(data, algos, **params)

        n_gains = len(model.coords['time_raw_gains'])
        n_vlt = len(model.coords['time_raw_vlt'])
        n_algos = model.n_algos
        resample_vars = {
            'log_vlt_time_raw': lambda: np.random.randn(n_algos, n_vlt),
            'gains_time_raw': lambda: np.random.randn(n_algos, n_gains),
        }
        compute_vars = ['mu', 'vlt']
        delete_vars = ['gains_time', 'log_vlt']
        input_vars = [var for var in self.trace.data_vars
                      if not var.startswith('_') and var not in delete_vars]
        outputs = [getattr(model.model, var) for var in compute_vars]
        inputs = [getattr(model.model, var) for var in input_vars]
        vals_func = theano.function(inputs, outputs, on_unused_input='ignore')

        predictions = []
        for (chain, sample), point in self._points():
            for var, draw in resample_vars.items():
                point[var] = draw()
            point = {var: point[var] for var in input_vars}
            mu, sd = vals_func(**point)
            returns = stats.norm(loc=mu, scale=sd).rvs().T
            returns = pd.DataFrame(returns, index=index, columns=columns)
            predictions.append(returns)
        return predictions


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
        seed = int(random.getrandbits(32))
    if 'random_seed' in sampler_args:
        raise ValueError('Can not specify `random_seed`.')
    sampler_args['random_seed'] = int(seed)

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
        ['log_gains_sd', 'log_gains_time_sd_sd']),
    'exponential': (
        'trace-exponential',
        ['log_gains_sd', 'log_gains_time_sd_sd']
    )
}


_DEFAULT_SHRINKAGE = {
    'default': (
        'trace-exponential',
        {
            'log_gains_sd': (-3.5, 0.2),
            'log_gains_time_sd_sd': (-3, 1),
        }
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

    if shrinkage in _TRACE_PARAM_NAMES:
        shrinkage, param_names = _TRACE_PARAM_NAMES[shrinkage]
    elif shrinkage in _DEFAULT_SHRINKAGE:
        warnings.warn('The default shrinkage is only a preview. The values '
                      '*will* change in the future.')
        shrinkage, param_defaults = _DEFAULT_SHRINKAGE[shrinkage]
        param_names = param_defaults.keys()
        for name in param_defaults:
            mu, sd = param_defaults[name]
            params.setdefault(name + '_trace_mu', float(mu))
            params.setdefault(name + '_trace_sd', float(sd))
    else:
        raise ValueError('Unknown shrinkage %s' % shrinkage)

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
