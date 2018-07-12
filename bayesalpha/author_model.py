""" Models the distribution of in-sample Sharpe ratios realized by authors. """

import random
import warnings
import json
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import LabelEncoder
import pymc3 as pm
from serialize import to_xarray
from bayesalpha._version import get_versions


class AuthorModelBuilder:
    def __init__(self, data):
        self.num_authors = data.meta_user_id.nunique()
        self.num_algos = data.meta_algorithm_id.nunique()
        # For num_backtests, nunique(), count() and len(data) should be the same
        self.num_backtests = data.meta_code_id.nunique()

        # Which algos correspond to which authors?
        df = (data[['meta_user_id', 'meta_algorithm_id']]
              .drop_duplicates(subset='meta_algorithm_id')
              .reset_index()
              .meta_user_id
              .astype(str))
        self.author_to_algo_encoding = LabelEncoder().fit_transform(df)

        # Which backtests correspond to which algos?
        df = data.meta_algorithm_id.astype(str)
        self.algo_to_backtest_encoding = LabelEncoder().fit_transform(df)

        # Which backtests correspond to which authors?
        df = data.meta_user_id.astype(str)
        self.author_to_backtest_encoding = LabelEncoder().fit_transform(df)

        self.model = self._build_model(data)

        self.coords = {
            'author':    data.meta_user_id.drop_duplicates().values,
            'algo':      data.meta_algorithm_id.drop_duplicates().values,
            'backtest':  data.meta_code_id.values
        }

        self.dims = {
            'mu_global':       (),
            'mu_author':       ('author', ),
            'mu_author_raw':   ('author', ),
            'mu_author_sd':    (),
            'sigma_author':    ('author', ),
            'sigma_author_sd': (),
            'mu_algo':         ('algo', ),
            'mu_algo_raw':     ('algo', ),
            'mu_algo_sd':      (),
            'sigma_algo':      ('algo', ),
            'sigma_algo_sd':   (),
            'mu_backtest':     ('backtest', ),
            'sigma_backtest':  ('backtest', ),
            'alpha_author':    ('author', ),
            'alpha_algo':      ('algo', )
        }

    def _build_model(self, data):
        """
        Build the entire author model (in one function). The model is
        sufficiently simple to specify entirely in one function.

        Parameters
        ----------
        data : pd.DataFrame
            Long-format DataFrame of in-sample Sharpe ratios (from user-run
            backtests), indexed by user, algorithm and code ID.
            Note that currently, backtests are deduplicated based on code id.

            See fit_authors for more information.
        """
        with pm.Model() as model:
            mu_global = pm.Normal('mu_global', mu=0, sd=3)

            mu_author_sd = pm.HalfNormal('mu_author_sd', sd=1)
            mu_author_raw = pm.Normal('mu_author_raw', mu=0, sd=1,
                                      shape=self.num_authors)
            mu_author = pm.Deterministic('mu_author',
                                         mu_author_sd * mu_author_raw)

            mu_algo_sd = pm.HalfNormal('mu_algo_sd', sd=1)
            mu_algo_raw = pm.Normal('mu_algo_raw', mu=0, sd=1,
                                    shape=self.num_algos)
            mu_algo = pm.Deterministic('mu_algo', mu_algo_sd * mu_algo_raw)

            mu_backtest = \
                pm.Deterministic('mu_backtest',
                                 mu_global
                                 + mu_author[self.author_to_backtest_encoding]
                                 + mu_algo[self.algo_to_backtest_encoding])

            sigma_author_sd = pm.HalfNormal('sigma_author_sd', sd=1)
            sigma_algo_sd = pm.HalfNormal('sigma_algo_sd', sd=1)

            sigma_author = pm.HalfNormal('sigma_author', sd=sigma_author_sd,
                                         shape=self.num_authors)
            sigma_algo = pm.HalfNormal('sigma_algo', sd=sigma_algo_sd,
                                       shape=self.num_algos)
            sigma_backtest = \
                pm.Deterministic(
                    'sigma_backtest',
                    np.sqrt(
                        np.square(
                            sigma_author[self.author_to_backtest_encoding]
                        )
                        + np.square(
                            sigma_algo[self.algo_to_backtest_encoding]
                        )
                    )
                )

            alpha_author = pm.Deterministic('alpha_author',
                                            mu_global + mu_author)

            alpha_algo = \
                pm.Deterministic('alpha_algo',
                                 mu_global
                                 + mu_author[self.author_to_algo_encoding]
                                 + mu_algo)

            sharpe = pm.Normal('sharpe',
                               mu=mu_backtest,
                               sd=sigma_backtest,
                               shape=self.num_backtests,
                               observed=data.perf_sharpe_ratio_is)

        return model


class FitResult:
    def __init__(self, trace):
        self._trace = trace

    def save(self, filename, group=None, **args):
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


def fit_authors(data,
                sampler_type='mcmc',
                sampler_args=None,
                seed=None,
                save_data=True,
                **params):
    """
    Fit author model to population of authors, with algos and backtests.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format DataFrame of in-sample Sharpe ratios (from user-run
        backtests), indexed by user, algorithm and code ID.
        Note that currently, backtests are deduplicated based on code id.
    ::
        meta_user_id   meta_algorithm_id   meta_code_id   perf_sharpe_ratio_is
    0   abcdef123456   ghijkl789123        abcdef000000   0.919407 
    1   abcdef123456   ghijkl789123        abcdef000001   1.129353 
    2   abcdef123456   ghijkl789123        abcdef000002   -0.005934

    sampler_type : str
        Whether to use Markov chain Monte Carlo or variational inference.
        Either 'mcmc' or 'vi'. Defaults to 'mcmc'.
    save_data : bool
        Whether to store the dataset in the result object.
    seed : int
        Seed for random number generation in PyMC3.
    """
    params = params.copy()

    if sampler_type not in {'mcmc', 'vi'}:
        raise ValueError("sampler_type not in {'mcmc', 'vi'}")

    if seed is None:
        seed = int(random.getrandbits(31))
    else:
        seed = int(seed)

    builder = AuthorModelBuilder(data)
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

    return FitResult(trace)


def _check_data(data):
    """
    Run basic sanity checks on the data set.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format DataFrame of in-sample Sharpe ratios (from user-run
        backtests), indexed by user, algorithm and code ID.
        Note that currently, backtests are deduplicated based on code id.

        See fit_authors for more information.
    """

    if data.meta_code_id.nunique() != data.shape[0]:
        warnings.warn('Data set contains duplicate backtests.')

    if (data.perf_sharpe_ratio_is > 5) | (data.perf_sharpe_ratio_is < -5):
        warnings.warn('Data set contains unrealistic Sharpes.')

    if (data.groupby('meta_algorithm_id')['perf_sharpe_ratio_is']
            .count() < 5).any():
        warnings.warn('Data set contains algorithms with fewer than 5 '
                      'backtests.')

    if (data.groupby('meta_user_id')['meta_algorithm_id'].nunique() < 5).any():
        warnings.warn('Data set contains users with fewer than 5 algorithms.')

    if pd.isnull(data).any().any():
        raise ValueError('Data set contains NaNs.')

    # FIXME remove this check once all feature factory features are debugged.
    if (data == -99999).any().any():
        raise ValueError('Data set contains -99999s.')
