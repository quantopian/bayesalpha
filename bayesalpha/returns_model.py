from functools import partial
import warnings

import numpy as np
from scipy import sparse, stats
import theano
import theano.sparse
import theano.tensor as tt
import pymc3 as pm
import json
import xarray as xr
from datetime import datetime
import random
import pandas as pd
import empyrical

from .base import BayesAlphaResult

try:
    import cvxpy
    if cvxpy.__version__.split('.') < ['1', '0', '0']:
        cvxpy = None
except ImportError:
    cvxpy = None

from bayesalpha.dists import (
    bspline_basis,
    GPExponential,
    NormalNonZero,
    ScaledSdMvNormalNonZero
)
from bayesalpha.dists import dot as sparse_dot
from bayesalpha.serialize import to_xarray
from bayesalpha._version import get_versions
import bayesalpha.plotting

_PARAM_DEFAULTS = {
    'shrinkage': 'skew-neg2-normal',
    'corr_type': 'diag',
    'gains_time': False,
}

RETURNS_MODEL_TYPE = 'returns-model'


class ReturnsModelBuilder(object):
    def __init__(self, data, algos, factors=None, predict=False,
                 gains_factors=None, **params):
        data = data.fillna(0.)
        self._predict = predict
        self.data = data
        # The build functions pop parameters they use
        self.params = params.copy()
        self.algos = algos
        if factors is None:
            factors = pd.DataFrame(index=data.index, columns=[])
            factors.columns.name = 'factor'

        if gains_factors is None:
            gains_factors = pd.DataFrame(index=data.index, columns=[])
            gains_factors.columns.name = 'gains_factor'

        # The build functions add items when appropriate
        self.coords = {
            'algo': data.columns,
            'time': data.index,
            'factor': factors.columns,
            'gains_factor': gains_factors.columns,
        }
        # The build functions add items when appropriate
        self.dims = {}
        self.n_algos = len(data.columns)
        self.n_time = len(data.index)
        self.n_factors = len(factors.columns)
        self.n_gains_factors = len(gains_factors.columns)
        self.factors = factors
        self.gains_factors = gains_factors

        if (not self._predict
                and factors is not None
                and not factors.index.equals(data.index)):
            raise ValueError('Factors must have the same index as data.')

        if (not self._predict
                and gains_factors is not None
                and not gains_factors.index.equals(data.index)):
            raise ValueError('Gains factors must have the same index as data.')

        self.model = pm.Model()
        with self.model:
            in_sample = self._build_in_sample()
            Bx_log_vlt, Bx_gains = self._build_splines()

            vlt = self._build_volatility(Bx_log_vlt)

            gains = self._build_gains_mu(in_sample)
            if self.params.pop('gains_time', False):
                gains_time = self._build_gains_time(Bx_gains)
                gains = gains + gains_time

            if len(gains_factors.columns) > 0 and not self._predict:
                factors_gains = self._build_gains_factors()
                gains = gains + factors_gains

            mu = gains * vlt
            if len(factors.columns) > 0 and not self._predict:
                factors_mu = self._build_returns_factors()
                mu = mu + factors_mu

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

    def _build_log_volatility_mean(self):
        self.corr_type = corr_type = self.params.pop('corr_type')
        k = self.n_algos
        if corr_type == 'diag':
            log_vlt_mu = pm.Normal('log_vlt_mu', mu=-6, sd=0.5, shape=k)
        elif corr_type == 'dense':
            vlt_mu_dist = pm.Lognormal.dist(mu=-2, sd=0.5, shape=k)
            chol_cov_packed = pm.LKJCholeskyCov(
                'chol_cov_packed_mu', n=k, eta=2, sd_dist=vlt_mu_dist)
            chol_cov = pm.expand_packed_triangular(k, chol_cov_packed) / np.exp(4)
            cov = tt.dot(chol_cov, chol_cov.T)
            variance_mu = tt.diag(cov)
            corr = cov / tt.sqrt(variance_mu[:, None] * variance_mu[None, :])
            pm.Deterministic('chol_cov_mu', chol_cov)
            pm.Deterministic('cov_mu', cov)
            pm.Deterministic('corr_mu', corr)
            # important, add new coordinate
            self.coords['algo_chol'] = pd.RangeIndex(k * (k + 1) // 2)
            self.coords['algo_'] = self.coords['algo']
            self.dims['chol_cov_packed_mu'] = ('algo_chol',)
            self.dims['cov_mu'] = ('algo', 'algo_')
            self.dims['corr_mu'] = ('algo', 'algo_')
            self.dims['chol_cov_mu'] = ('algo', 'algo_')
            log_vlt_mu = pm.Deterministic('log_vlt_mu', tt.log(variance_mu) / 2.)
        else:
            raise NotImplementedError
        self.dims['log_vlt_mu'] = ('algo',)
        return log_vlt_mu

    def _build_log_volatility_time(self):
        k = self.n_algos
        n_knots_vlt = len(self.coords['time_raw_vlt'])

        log_vlt_time_alpha = pm.HalfNormal('log_vlt_time_alpha', sd=0.1)
        log_vlt_time_sd = pm.HalfNormal('log_vlt_time_sd', sd=0.5, shape=k)
        self.dims['log_vlt_time_sd'] = ('algo',)
        log_vlt_time_raw = GPExponential(
            'log_vlt_time_raw', mu=0, alpha=log_vlt_time_alpha,
            sigma=1, shape=(k, n_knots_vlt))
        self.dims['log_vlt_time_raw'] = ('algo', 'time_raw_vlt')
        return log_vlt_time_sd[:, None] * log_vlt_time_raw

    def _build_volatility(self, Bx_log_vlt):
        log_vlt_mu = self._build_log_volatility_mean()
        log_vlt_time_raw = self._build_log_volatility_time()
        log_vlt_time = sparse_dot(Bx_log_vlt, log_vlt_time_raw.T).T
        log_vlt = log_vlt_time + log_vlt_mu[:, None]
        pm.Deterministic('log_vlt_time', log_vlt_time)
        pm.Deterministic('log_vlt', log_vlt)
        self.dims['log_vlt_time'] = ('algo', 'time')
        self.dims['log_vlt'] = ('algo', 'time')
        vlt = tt.exp(log_vlt)
        return vlt

    def _build_gains_mu(self, is_author_is):
        self.dims.update({
            'gains_theta': ('algo',),
            'gains_eta': ('algo',),
            'author_is': ('algo',),
            'author_is_raw': ('algo',),
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
            gains_sd = pm.HalfNormal('gains_sd', sd=0.1)
            pm.Deterministic('log_gains_sd', tt.log(gains_sd))
            gains_mu = pm.Normal('gains_mu', mu=0.05, sd=0.1)
            gains_raw = pm.Normal('gains_raw', shape=k)

            author_is = pm.HalfNormal('author_is', shape=k, sd=0.1)
            gains = pm.Deterministic('gains', gains_sd * gains_raw + gains_mu)
            gains_all = gains[None, :] + author_is[None, :] * is_author_is
        elif shrinkage == 'skew-neg2-normal':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.1)
            pm.Deterministic('log_gains_sd', tt.log(gains_sd))
            gains_mu = pm.Normal('gains_mu', sd=0.1)
            gains_raw = pm.SkewNormal(
                'gains_raw', sd=1, mu=0, alpha=-4, shape=k)

            author_is = pm.Normal('author_is', shape=k, sd=0.4, mu=0.0)
            gains = pm.Deterministic('gains', gains_sd * gains_raw + gains_mu)
            gains_all = (
                (1 - is_author_is) * gains[None, :]
                + author_is[None, :] * is_author_is
            )
        elif shrinkage == 'skew-normal':
            gains_sd = pm.HalfNormal('gains_sd', sd=0.1)
            pm.Deterministic('log_gains_sd', tt.log(gains_sd))
            gains_alpha = pm.Normal('gains_alpha', sd=0.3)
            gains_mu = pm.Normal('gains_mu', mu=0.05, sd=0.1)
            gains_raw = pm.SkewNormal(
                'gains_raw', sd=1, mu=0, alpha=gains_alpha, shape=k)

            author_is = pm.Normal('author_is', shape=k)
            gains = pm.Deterministic('gains', gains_sd * gains_raw + gains_mu)
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
        gains_time_sd_raw = pm.HalfNormal('gains_time_sd_raw', shape=k)
        gains_time_sd = pm.Deterministic(
            'gains_time_sd', gains_time_sd_sd * gains_time_sd_raw)
        gains_time_raw = GPExponential(
            'gains_time_raw', mu=0, alpha=gains_time_alpha,
            sigma=1, shape=(k, n_knots_gains))
        gains_time = gains_time_sd[:, None] * gains_time_raw
        gains_time = sparse_dot(Bx_gains, gains_time.T).T

        pm.Deterministic('gains_time', gains_time)
        return gains_time

    def _build_likelihood(self, mu, sd, observed):
        corr_type = self.corr_type
        if corr_type == 'diag':
            NormalNonZero('y', mu=mu, sd=sd, observed=observed)
        elif corr_type == 'dense':
            # mu, sd  --`shape`-- (algo, time)
            # mv needs (time, algo)
            ScaledSdMvNormalNonZero(
                'y', mu=mu.T,
                chol=self.model.named_vars['chol_cov_mu'],
                scale_sd=tt.exp(self.model.named_vars['log_vlt_time'].T),
                observed=observed.T)
        else:
            raise NotImplementedError
        if self._predict:
            self.dims['mu'] = ('algo', 'time')
            pm.Deterministic('mu', mu)
        if self._predict:
            self.dims['vlt'] = ('algo', 'time')
            pm.Deterministic('vlt', sd)

    def _build_gains_factors(self):
        self.dims.update({
            'gains_factor_algo': ('gains_factor', 'algo'),
            'gains_factor_algo_raw': ('gains_factor', 'algo'),
            'gains_factor_algo_sd': ('gains_factor',),
        })
        gains_factors = self.gains_factors
        n_algos, n_gains_factors = self.n_algos, self.n_gains_factors
        sd = pm.HalfNormal('gains_factor_algo_sd', sd=0.4,
                           shape=n_gains_factors)
        raw = pm.StudentT('gains_factor_algo_raw', nu=7, mu=0, sd=1,
                          shape=(n_gains_factors, n_algos))
        vals = sd[:, None] * raw
        pm.Deterministic('gains_factor_algo', vals)
        return (vals[:, None, :] * gains_factors.values.T[:, :, None]).sum(0).T

    def _build_returns_factors(self):
        self.dims.update({
            'factor_algo': ('factor', 'algo'),
        })
        factors = self.factors
        n_algos, n_factors = self.n_algos, self.n_factors
        factor_algo = pm.StudentT('factor_algo', nu=3, mu=0, sd=2,
                                  shape=(n_factors, n_algos))
        return (factor_algo[:, None, :]
                * factors.values.T[:, :, None]).sum(0).T

    def make_predict_function(self, factor_scale_halflife=None):
        if not self._predict:
            raise ValueError('Model was not built for predictions.')

        if factor_scale_halflife is not None:
            factor_scales = (self.factors
                .ewm(halflife=factor_scale_halflife)
                .std()
                .iloc[-1])

        n_gains = len(self.coords['time_raw_gains'])
        n_vlt = len(self.coords['time_raw_vlt'])
        n_algos = self.n_algos
        resample_vars = {
            'log_vlt_time_raw': lambda: np.random.randn(n_algos, n_vlt),
            'gains_time_raw': lambda: np.random.randn(n_algos, n_gains),
        }

        if self.corr_type == 'diag':
            compute_vars = ['mu', 'vlt']
        elif self.corr_type == 'dense':
            compute_vars = ['mu', 'log_vlt_time', 'chol_cov_mu']
        else:
            raise NotImplementedError('Unkown correlation type.')

        delete_vars = ['gains_time', 'log_vlt', 'mu', 'vlt', 'log_vlt_time']
        input_vars = [var.name for var in self.model.unobserved_RVs
                      if (not var.name.endswith('_')
                          and var.name not in delete_vars)]
        outputs = [getattr(self.model, var) for var in compute_vars]
        inputs = [getattr(self.model, var) for var in input_vars]
        # downcast inputs if needed
        vals_func = theano.function(inputs, outputs, on_unused_input='ignore',
                                    allow_input_downcast=True)

        algos = self.coords['algo']
        factors = self.coords['factor']
        time = self.coords['time']

        def predict(point):
            if factor_scale_halflife is not None:
                factor_exposures = point['factor_algo']

            for var, draw in resample_vars.items():
                point[var] = draw()
            point = {var: point[var] for var in input_vars}

            if self.corr_type == 'diag':
                mu, sd = vals_func(**point)
                returns = stats.norm(loc=mu, scale=sd).rvs()
            elif self.corr_type == 'dense':
                mu, log_vlt_time, chol = vals_func(**point)
                returns = np.random.randn(len(algos), len(time))
                returns = np.dot(chol, returns)
                returns[...] *= np.exp(log_vlt_time)
                returns[...] += mu

            if factor_scale_halflife is not None and len(factor_scales) > 0:
                factor_rets = np.random.randn(len(factor_scales), len(time))
                factor_rets = factor_rets * factor_scales[:, None]
                factor_rets = factor_rets[None, :, :] * factor_exposures.T[:, :, None]
                factor_rets = factor_rets.sum(1)
                returns[...] += factor_rets
                exposures = xr.DataArray(factor_exposures, coords=[factors, algos])
            else:
                exposures = None
                
            returns = xr.DataArray(returns, coords=[algos, time])
            
            return xr.Dataset({'returns': returns, 'exposures': exposures})

        return predict


class ReturnsModelResult(BayesAlphaResult):
    def plot_prob(self,
                  algos=None,
                  ax=None,
                  sort=True,
                  rope=False,
                  rope_upper=.05,
                  rope_lower=None,
                  title=None):
        if rope:
            prob_func = partial(self.gains_rope, rope_upper, lower=rope_lower)
        else:
            prob_func = self.gains_pos_prob

        if algos is not None:
            vals = prob_func().loc[algos]
        else:
            vals = prob_func()

        xlabel = 'P(gains ~ 0)' if rope else 'P(gains > 0)'

        ax = bayesalpha.plotting.plot_horizontal_dots(
            vals,
            sort=sort,
            ax=ax,
            xlabel=xlabel,
            title=title
        )

        return ax

    def plot_corr(self, algos=None, corr_threshold=.33,
                  ax=None, cmap=None, **heatmap_kwargs):
        corr = self.trace['corr_mu']
        if algos is not None:
            corr = corr.loc[dict(algo=algos, algo_=algos)]
        return bayesalpha.plotting.plot_correlations(
            corr_xarray=corr, ax=ax, corr_threshold=corr_threshold,
            cmap=cmap, **heatmap_kwargs)

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

    def _points(self, include_transformed=True):
        for chain in self.trace.chain:
            for sample in self.trace.sample:
                vals = self.trace.sel(chain=chain, sample=sample)
                data = {}
                for var in self.trace.data_vars:
                    if not include_transformed and var.startswith('_'):
                        continue
                    data[var] = vals[var].values
                yield (chain, sample), data

    def _random_point_iter(self, include_transformed=True):
        while True:
            chain = np.random.randint(len(self.trace.chain))
            sample = np.random.randint(len(self.trace.sample))
            data = {}
            vals = self.trace.isel(chain=chain, sample=sample)
            for var in self.trace.data_vars:
                if not include_transformed and var.startswith('_'):
                    continue
                data[var] = vals[var].values
            yield data

    def rebuild_model(self, data=None, algos=None, factors=None,
                      gains_factors=None, **extra_params):
        """Return a ReturnsModelBuilder that recreates the original model."""
        if data is None:
            data = self.trace._data.to_pandas().copy()
        if algos is None:
            algos = self.trace._algos.to_pandas().copy()
        if factors is None:
            factors = self.trace._factors.to_pandas().copy()
        if gains_factors is None:
            gains_factors = self.trace._gains_factors.to_pandas().copy()
        params = self.params.copy()
        params.update(extra_params)
        return ReturnsModelBuilder(data, algos, factors=factors,
                                   gains_factors=gains_factors, **params)

    def _make_prediction_model(self, n_days):
        start = pd.Timestamp(self.trace.time[-1].values)
        index = pd.date_range(start, periods=n_days, freq='B', name='time')
        columns = self.trace.algo

        data = pd.DataFrame(index=index, columns=columns)
        data.values[...] = 0.
        algos = self.trace.algo.to_pandas().copy().to_frame()
        algos['created_at'] = start
        return self.rebuild_model(data, algos, predict=True)

    def predict(self, n_days, n_repl=None, factor_scale_halflife=None):
        model = self._make_prediction_model(n_days)
        predict_func = model.make_predict_function(factor_scale_halflife)
        coords = [self.trace.chain, self.trace.sample,
                  self.trace.algo, model.coords['time']]
        coords_exposures = [self.trace.chain, self.trace.sample,
                            self.trace.factor, self.trace.algo]
        if n_repl is not None:
            repl_coord = pd.RangeIndex(n_repl, name='sim_repl')
            coords.append(repl_coord)
            coords_exposures.append(repl_coord)
        shape = [len(vals) for vals in coords]

        returns_data = np.zeros(shape)
        exposure_data = np.zeros([len(v) for v in coords_exposures])
        returns = xr.DataArray(returns_data, coords=coords)
        exposures = xr.DataArray(exposure_data, coords=coords_exposures)
        for (chain, sample), point in self._points(include_transformed=False):
            if n_repl is None:
                prediction = predict_func(point)
                returns.loc[chain, sample, :, :] = prediction.returns
                exposures.loc[chain, sample, :, :] = prediction.exposures
            else:
                for repl in repl_coord:
                    prediction = predict_func(point)
                    returns.loc[chain, sample, :, :, repl] = prediction.returns
                    exposures.loc[chain, sample, :, :, repl] = prediction.exposures
        return xr.Dataset({'returns': returns, 'exposures': exposures})

    def predict_value(self, n_days, n_repl=None, factor_scale_halflife=None):
        model = self._make_prediction_model(n_days)
        predict_func = model.make_predict_function(factor_scale_halflife)
        coords = [self.trace.chain, self.trace.sample, self.trace.algo]
        if n_repl is not None:
            repl_coord = pd.RangeIndex(n_repl, name='sim_repl')
            coords.append(repl_coord)
        shape = [len(vals) for vals in coords]

        prediction_data = np.zeros(shape)
        predictions = xr.DataArray(prediction_data, coords=coords)
        for (chain, sample), point in self._points(include_transformed=False):
            if n_repl is None:
                returns = predict_func(point).to_pandas().T
                cum_returns = returns.apply(empyrical.cum_returns_final)
                predictions.loc[chain, sample, :] = cum_returns
            else:
                for repl in repl_coord:
                    returns = predict_func(point).to_pandas().T
                    cum_returns = returns.apply(empyrical.cum_returns_final)
                    predictions.loc[chain, sample, :, repl] = cum_returns
        return predictions

    def prediction_iter(self, n_days):
        model = self._make_prediction_model(n_days)
        predict_func = model.make_predict_function()
        for point in self._random_point_iter(include_transformed=False):
            yield predict_func(point)


class Optimizer(object):
    def __init__(self, predictions, utility='isoelastic', lmda=None,
                 factor_penalty=None, max_weights=None, exposure_limit=None,
                 exposure_penalty=None):
        """Compute a portfolio based on model predictions.

        Parameters
        ----------
        predictions : xr.Dataset
            Predictions as returned by fit.predict_value
        utility : ['isoelastic', 'exp'], default='isoelastic'
            The utility function to use.
        lmda : float
            Risk aversion parameter. This value can be overridden
            by passing a different value to `solve`.
        factor_penalty : float
            Add a penalty during the optimization for portfolios that have
            exposure to risk factors. This uses the estimates of risk exposure
            from the regression in bayesalpha. High values mean that we are
            willing to take hits on the predicted portfolio in order to
            decrease risk exposure.
        max_weights : list
            A maximum weight for each algo.
        exposure_limit : float
            A hard limit for risk exposures of each weighted algo in the
            portfolio. This uses the position based risk exposure passed in as
            `predictions.position_exposures`, and limits the maximum risk
            exposure of each algo over that time period.
        exposure_penalty : float
            This also uses the position based exposure, but adds a quadratic
            penalty term during optimization instead of a hard limit.
        """
        if cvxpy is None:
            raise RuntimeError('Optimization requires cvxpy>=1.0')
        self._predictions = predictions
        self._problem = self._build_problem(lmda, utility, factor_penalty,
                                            exposure_limit, exposure_penalty)
        if max_weights is None:
            max_weights = [1] * len(predictions.algo)
        self._max_weights = max_weights

    def _build_problem(self, lmda_vals, utility, factor_penalty=None,
                       exposure_limit=None, exposure_penalty=None):
        n_predict = (len(self._predictions.chain)
                     * len(self._predictions.sample)
                     * len(self._predictions.sim_repl))
        n_algos = len(self._predictions.algo)
        n_factors = len(self._predictions.factor)
        lmda = cvxpy.Parameter(name='lambda', nonneg=True)
        returns = cvxpy.Parameter(shape=(n_predict, n_algos), name='returns')
        max_weights = cvxpy.Parameter(shape=(n_algos), name='max_weights')
        weights = cvxpy.Variable(shape=(n_algos,), name='weights')
        portfolio_returns = returns * weights
        if utility == 'exp':
            risk = cvxpy.log_sum_exp(-lmda * portfolio_returns)
        elif utility == 'isoelastic':
            risk = cvxpy.log_sum_exp(-lmda * cvxpy.log(portfolio_returns))
        else:
            raise ValueError('Unknown utility: %s' % utility)

        if factor_penalty is not None:
            penalty = cvxpy.Parameter(shape=(), name='factor_penalty', nonneg=True)
            self._factor_penalty_p = penalty
            for i in range(n_factors):
                exposures = cvxpy.Parameter(shape=(n_predict, n_algos),
                                            name='exposures_%s' % i)
                exposures.value = self._predictions.exposures.isel(factor=i).stack(
                    prediction=('chain', 'sample', 'sim_repl')).values.T
                risk_factor = cvxpy.sum_squares(exposures * weights)
                risk = risk + penalty * risk_factor

        if exposure_penalty is not None:
            penalty = cvxpy.Parameter(shape=(), name='exposure_penalty', nonneg=True)
            self._exposure_penalty_p = penalty
            exposure_data = self._predictions.position_exposures
            n_history = len(exposure_data.time_hist)
            exposures = cvxpy.Parameter(shape=(n_history, n_algos),
                                        name='position_exposures')
            exposures.value = exposure_data.values
            risk_factor = cvxpy.sum_squares(exposures * weights)
            risk = risk + penalty * risk_factor * n_predict / n_history

        constraints = [cvxpy.sum(weights) == 1, weights >= 0, weights <= max_weights]
        if exposure_limit is not None:
            limit = cvxpy.Parameter(name='exposure_limit', nonneg=True)
            self._exposure_limit = limit
            limit.value = exposure_limit
            exposures_lower = cvxpy.Parameter(shape=(n_algos,), name='exposures_lower')
            exposures_upper = cvxpy.Parameter(shape=(n_algos,), name='exposures_upper')
            exposure_data = self._predictions.position_exposures
            exposures_lower.value = exposure_data.sel(quantile='lower').values
            exposures_upper.value = exposure_data.sel(quantile='upper').values
            lower = cvxpy.sum(weights * exposures_lower) >= -limit
            upper = cvxpy.sum(weights * exposures_upper) <= limit
            constraints.extend([lower, upper])

        problem = cvxpy.Problem(cvxpy.Minimize(risk), constraints)

        if lmda_vals is not None:
            lmda.value = lmda_vals
        predictions = self._predictions.cum_final.stack(
            prediction=('chain', 'sample', 'sim_repl'))
        # +1 because we want the final wealth, when we start with
        # a unit of money.
        returns.value = predictions.values.T + 1

        self._lmda_p = lmda
        self._factor_weights_p = None
        self._weights_v = weights
        self._max_weights_v = max_weights
        return problem

    def solve(self, lmda=None, factor_penalty=None, max_weights=None,
              exposure_limit=None, exposure_penalty=None, **kwargs):
        """Find the optimal weights for the portfolio."""
        if lmda is not None:
            self._lmda_p.value = lmda
        if exposure_penalty is not None:
            self._exposure_penalty_p.value = exposure_penalty
        if factor_penalty is not None:
            self._factor_penalty_p.value = factor_penalty
        if exposure_limit is not None:
            self._exposure_limit.value = exposure_limit
        if max_weights is not None:
            self._max_weights_v.value = max_weights
        else:
            self._max_weights_v.value = np.array(self._max_weights)
        self._problem.solve(**kwargs)
        if self._problem.status != 'optimal':
            raise ValueError('Optimization did not converge.')
        weights = self._weights_v.value.ravel().copy()
        algos = self._predictions.algo
        return xr.DataArray(weights, coords=[algos], name='weights')


def fit_returns_population(data, algos=None, sampler_args=None, save_data=True,
                           seed=None, factors=None, gains_factors=None,
                           sampler_type='mcmc', **params):
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
    if sampler_type not in {'mcmc', 'vi'}:
        raise ValueError("sampler_type not in {'mcmc', 'vi'}")
    _check_data(data)
    params_ = _PARAM_DEFAULTS.copy()
    params_.update(params)
    params = params_

    if algos is None:
        algos = pd.DataFrame(columns=['created_at'], index=data.columns)
        algos['created_at'] = data.index[0]

    if sampler_args is None:
        sampler_args = {}
    else:
        sampler_args = sampler_args.copy()
    if seed is None:
        seed = int(random.getrandbits(31))
    else:
        seed = int(seed)
    if 'random_seed' in sampler_args:
        raise ValueError('Can not specify `random_seed`.')
    sampler_args['random_seed'] = seed

    builder = ReturnsModelBuilder(data, algos, factors=factors,
                                  gains_factors=gains_factors, **params)
    model, coords, dims = builder.model, builder.coords, builder.dims

    timestamp = datetime.isoformat(datetime.now())
    with model:
        args = {} if sampler_args is None else sampler_args
        with warnings.catch_warnings(record=True) as warns:
            if sampler_type == 'mcmc':
                trace = pm.sample(**args)
            else:
                trace = pm.fit(**args).sample(args.get('draws', 500))
    if warns:
        warnings.warn('Problems during sampling. Inspect `result.warnings`.')
    trace = to_xarray(trace, coords, dims)
    trace.attrs['params'] = json.dumps(params)
    trace.attrs['timestamp'] = timestamp
    trace.attrs['warnings'] = json.dumps([str(warn) for warn in warns])
    trace.attrs['seed'] = seed
    trace.attrs['model-version'] = get_versions()['version']
    trace.attrs['model-type'] = RETURNS_MODEL_TYPE

    if save_data:
        trace.coords['algodata'] = algos.columns
        trace['_data'] = (('time', 'algo'), data)
        try:
            trace['_algos'] = (('algo', 'algodata'), algos.loc[data.columns])
        except ValueError:
            warnings.warn('Could not save algo metadata, skipping.')
        try:
            trace['_factors'] = (('time', 'factor'), builder.factors)
        except ValueError:
            warnings.warn('Could not save algo metadata, skipping.')
        try:
            trace['_gains_factors'] = (('time', 'gains_factor'),
                                       builder.gains_factors)
        except ValueError:
            warnings.warn('Could not save algo metadata, skipping.')
    return ReturnsModelResult(trace)


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


def fit_returns_single(data, algos=None, population_fit=None,
                       sampler_args=None, seed=None, factors=None, **params):
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
        The result of a previous model fit using `fit_returns_population`. If
        this is not specified, all necessary parameters have to be specified as
        keyword arguments.
    sampler_args : dict
        Additional arguments for `pm.sample`
    seed : int
        Seed for random numbers during sampling.
    """
    params = params.copy()

    if population_fit is None:
        trace_shrinkage = None
        trace_corr = None
    else:
        trace_shrinkage = population_fit.params['shrinkage']
        trace_corr = population_fit.params['corr']

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

    fit = fit_returns_population(data, algos=algos, sampler_args=sampler_args,
                                 seed=seed, shrinkage=shrinkage,
                                 factors=factors, **params)
    if population_fit is not None:
        parent = population_fit.trace
        fit.trace.attrs['parent-params'] = parent.attrs['params']
        fit.trace.attrs['parent-seed'] = parent.attrs['seed']
        fit.trace.attrs['parent-version'] = parent.attrs['model-version']
        fit.trace.attrs['parent-type'] = RETURNS_MODEL_TYPE
        fit.trace.attrs['parent-id'] = population_fit.id
    return fit


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
