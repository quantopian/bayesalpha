""" Models the distribution of in-sample Sharpe ratios realized by authors. """

import random
import warnings
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.covariance import LedoitWolf
import pymc3 as pm
import theano.tensor as tt
import xarray as xr
from .serialize import to_xarray
from ._version import get_versions
from .base import BayesAlphaResult

AUTHOR_MODEL_TYPE = 'author-model'
APPROX_BDAYS_PER_YEAR = 252


class AuthorModelBuilder(object):
    """ Class to build the author model.  """

    def __init__(self, sharpes, returns):
        """
        Initialize AuthorModelBuilder object.

        Parameters
        ----------
        sharpes : pd.DataFrame
            Long-format DataFrame of in-sample Sharpe ratios (from user-run
            backtests), indexed by user, algorithm and code ID.
            Note that currently, backtests are deduplicated based on code id.
            See fit_authors for more information.
        """
        self.num_authors = sharpes.meta_user_id.nunique()
        self.num_algos = sharpes.meta_algorithm_id.nunique()
        # For num_backtests, nunique() and count() should be the same
        self.num_backtests = sharpes.meta_code_id.nunique()

        # Which algos correspond to which authors?
        df = (sharpes.loc[:, ['meta_user_id', 'meta_algorithm_id']]
              .drop_duplicates(subset='meta_algorithm_id', keep='first')
              .reset_index()
              .meta_user_id
              .astype(str))
        self.author_to_algo_encoding = LabelEncoder().fit_transform(df)

        # Which backtests correspond to which algos?
        df = sharpes.meta_algorithm_id.astype(str)
        self.algo_to_backtest_encoding = LabelEncoder().fit_transform(df)

        # Which backtests correspond to which authors?
        df = sharpes.meta_user_id.astype(str)
        self.author_to_backtest_encoding = LabelEncoder().fit_transform(df)

        # Construct correlation matrix.
        # 0 is a better estimate for mean returns than the sample mean!
        returns_ = returns / returns.std()
        self.corr = LedoitWolf(assume_centered=True).fit(returns_).covariance_

        self.model = self._build_model(sharpes, self.corr)

        self.coords = {
            'meta_user_id':      sharpes.meta_user_id.drop_duplicates().values,
            'meta_algorithm_id': sharpes.meta_algorithm_id.drop_duplicates().values,
            'meta_code_id':      sharpes.meta_code_id.values
        }

        self.dims = {
            'mu_global':       (),
            'mu_author':       ('meta_user_id', ),
            'mu_author_raw':   ('meta_user_id', ),
            'mu_author_sd':    (),
            'mu_algo':         ('meta_algorithm_id', ),
            'mu_algo_raw':     ('meta_algorithm_id', ),
            'mu_algo_sd':      (),
            'mu_backtest':     ('meta_code_id', ),
            'sigma_backtest':  ('meta_code_id', ),
            'alpha_author':    ('meta_user_id', ),
            'alpha_algo':      ('meta_algorithm_id', )
        }

    def _build_model(self, sharpes, corr):
        """
        Build the entire author model (in one function). The model is
        sufficiently simple to specify entirely in one function.

        Parameters
        ----------
        sharpes : pd.DataFrame
            Long-format DataFrame of in-sample Sharpe ratios (from user-run
            backtests), indexed by user, algorithm and code ID.
            Note that currently, backtests are deduplicated based on code id.
            See fit_authors for more information.
        corr : np.ndarray
            Correlation matrix of returns streams (from backtests), estimated
            using Ledoit-Wolf shrinkage.
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

            sigma_backtest = pm.Deterministic(
                'sigma_backtest',
                tt.sqrt(APPROX_BDAYS_PER_YEAR / sharpes.meta_trading_days)
            )

            cov = corr * sigma_backtest[:, None] * sigma_backtest[None, :]

            alpha_author = pm.Deterministic('alpha_author',
                                            mu_global + mu_author)

            alpha_algo = \
                pm.Deterministic('alpha_algo',
                                 mu_global
                                 + mu_author[self.author_to_algo_encoding]
                                 + mu_algo)

            sharpe = pm.MvNormal('sharpe',
                                 mu=mu_backtest,
                                 cov=cov,
                                 shape=self.num_backtests,
                                 observed=sharpes.sharpe_ratio)

        return model


class AuthorModelResult(BayesAlphaResult):
    def rebuild_model(self, sharpes=None, returns=None):
        """ Return an AuthorModelBuilder that recreates the original model. """
        if sharpes is None:
            sharpes = (self.trace
                       ._sharpes
                       .to_pandas()
                       .reset_index()
                       .copy())

        if returns is None:
            returns = (self.trace
                       ._returns
                       .to_pandas()
                       .copy())

        return AuthorModelBuilder(sharpes, returns)


def fit_authors(sharpes,
                returns,
                sampler_type='mcmc',
                sampler_args=None,
                seed=None,
                save_data=True,
                **params):
    """
    Fit author model to population of authors, with algos and backtests.

    Parameters
    ----------
    sharpes : pd.DataFrame
        Long-format DataFrame of in-sample Sharpe ratios (from user-run
        backtests), indexed by user, algorithm and code ID.
        Note that currently, backtests are deduplicated based on code id.
    ::
       meta_user_id  meta_algorithm_id  meta_code_id  meta_trading_days  sharpe_ratio
    0  abcdef123456  ghijkl789123       abcdef000000  136                0.919407
    1  abcdef123456  ghijkl789123       abcdef000001  271                1.129353
    2  abcdef123456  ghijkl789123       abcdef000002  229                -0.005934

    returns : pd.DataFrame
        Wide-format DataFrame of in-sample returns of user-run backtests,
        indexed by time. Columns are code ids, rows are time (the format of
        time does not matter).
    ::
                  abcd1234      efgh5678      ijkl9123
    2013-06-03   -0.000326      0.002815      0.002110
    2013-06-04    0.000326     -0.000135     -0.001211
    2013-06-05    0.000326      0.001918      0.002911

    sampler_type : str
        Whether to use Markov chain Monte Carlo or variational inference.
        Either 'mcmc' or 'vi'. Defaults to 'mcmc'.
    sampler_args : dict
        Additional parameters for `pm.sample`.
    save_data : bool
        Whether to store the dataset in the result object.
    seed : int
        Seed for random number generation in PyMC3.
    """
    if params:
        raise ValueError('Unnecessary kwargs passed to fit_authors.')

    if sampler_type not in {'mcmc', 'vi'}:
        raise ValueError("sampler_type not in {'mcmc', 'vi'}")

    # Check data
    _check_data(sharpes, returns)

    if seed is None:
        seed = int(random.getrandbits(31))
    else:
        seed = int(seed)

    builder = AuthorModelBuilder(sharpes, returns)
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
    # Author model takes no parameters, so this will always be empty.
    trace.attrs['params'] = json.dumps(params)
    trace.attrs['timestamp'] = timestamp
    trace.attrs['warnings'] = json.dumps([str(warn) for warn in warns])
    trace.attrs['seed'] = seed
    trace.attrs['model-version'] = get_versions()['version']
    trace.attrs['model-type'] = AUTHOR_MODEL_TYPE

    if save_data:
        # Store the data in long format to avoid creating more dimensions
        trace['_sharpes'] = xr.DataArray(sharpes, dims=['sharpes_index',
                                                        'sharpes_columns'])
        trace['_returns'] = xr.DataArray(returns, dims['returns_index',
                                                       'returns_columns'])

    return AuthorModelResult(trace)


def _check_data(sharpes, returns):
    """
    Run basic sanity checks on the data set.

    Parameters
    ----------
    sharpes : pd.DataFrame
        Long-format DataFrame of in-sample Sharpe ratios (from user-run
        backtests), indexed by user, algorithm and code ID.
        Note that currently, backtests are deduplicated based on code id.
        See fit_authors for more information.
    returns : pd.DataFrame
        Wide-format DataFrame of in-sample returns of user-run backtests,
        indexed by time. Columns are code ids, rows are time (the format of
        time does not matter).
        See fit_authors for more information.
    """

    # FIXME deduplicating based on code id is not perfect. Ideally we would
    # deduplicate on backtest id.
    if sharpes.meta_code_id.nunique() != sharpes.shape[0]:
        warnings.warn('Data set contains duplicate backtests.')

    if (sharpes.groupby('meta_algorithm_id')['sharpe_ratio']
            .count() < 5).any():
        warnings.warn('Data set contains algorithms with fewer than 5 '
                      'backtests.')

    if (sharpes.groupby('meta_user_id')['meta_algorithm_id'].nunique() < 5).any():
        warnings.warn('Data set contains users with fewer than 5 algorithms.')

    if ((sharpes.sharpe_ratio > 20)
            | (sharpes.sharpe_ratio < -20)).any():
        raise ValueError('`sharpes` contains unrealistic values: greater than '
                         '20 in magnitude.')

    if pd.isnull(sharpes).any().any():
        raise ValueError('`sharpes` contains NaNs.')

    # FIXME remove this check once all feature factory features are debugged.
    if (sharpes == -99999).any().any():
        raise ValueError('`sharpes` contains -99999s.')

    if pd.isnull(returns).any().any():
        raise ValueError('`returns` contains NaNs.')

    if returns.columns.duplicated().any():
        raise ValueError('`returns` contains duplicated code ids.')

    if len(sharpes.meta_code_id) != len(returns.columns):
        raise ValueError('`sharpes` and `returns` are different lengths.')

    if not set(sharpes.meta_code_id) == set(returns.columns):
        raise ValueError('`sharpes` and `returns` are the same length, but '
                         'contain different code ids.')

    if not (sharpes.meta_code_id == returns.columns).all():
        raise ValueError('`sharpes` and `returns` contain the same code ids, '
                         'but are ordered differently.')
