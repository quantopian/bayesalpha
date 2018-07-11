import numpy as np
import pandas as pd
import random
import xarray as xr
from sklearn.preprocessing import LabelEncoder
import warnings
import pymc3 as pm


class ModelBuilder:
    def __init__(self, data):
        self.num_authors = data.meta_user_id.nunique()
        self.num_algos = data.meta_algorithm_id.nunique()
        # num_backtests: nunique(), count() and len(data) should be the same
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
                        np.square(sigma_author[self.author_to_backtest_encoding])
                        + np.square(sigma_algo[self.algo_to_backtest_encoding])
                    )
                )

            alpha_author = pm.Deterministic('alpha_author',
                                            mu_global + mu_author)

            alpha_algo = pm.Deterministic('alpha_algo',
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


def fit_authors(data, sampler_type='mcmc', sampler_args=None, seed=None,
        save_data=True):
    """
    Fit author model to population of authors, with algos and backtests.

    Parameters
    ----------
    data : pd.DataFrame
        In-sample Sharpe ratios of backtests, indexed by user, algorithm
        and code ID. Long format.
        Note that currently, backtests are deduplicated based on code id.
    ::
             meta_user_id    meta_algorithm_id    meta_code_id    perf_sharpe_ratio_is
        0    abcdef123456    ghijkl789123         abcdef000000    0.919407 
        1    abcdef123456    ghijkl789123         abcdef000001    1.129353 
        2    abcdef123456    ghijkl789123         abcdef000002   -0.005934
    sampler_type : str
        Either 'mcmc' or 'vi'
    save_data : bool
        Whether to store the dataset in the result object.
    seed : int
        Seed for random number generation in PyMC3.
    """
    if sampler_type not in {'mcmc', 'vi'}:
        raise ValueError("sampler_type not in {'mcmc', 'vi'}")

    if seed is None:
        seed = int(random.getrandbits(31))
    else:
        seed = int(seed)

    builder = ModelBuilder(data)
    model, coords, dims = builder.model, builder.coords, builder.dims

    with model:
        args = {} if sampler_args is None else sampler_args
        with warnings.catch_warnings(record=True) as warns:
            if sampler_type == 'mcmc':
                trace = pm.sample(**args)
            else:
                trace = pm.fit(**args).sample(args.get('draws', 500))


def _check_data(data):
    # TODO check for sufficient data: more than 5 algos per author, more than 5
    # backtests per algo.

    if data.meta_code_id.nunique() != data.shape[0]:
        warnings.warn('Data set contains duplicate backtests.')

    if (data.perf_sharpe_ratio_is > 5) | (data.perf_sharpe_ratio_is < -5):
        warnings.warn('Data set contains unrealistic Sharpes. Proceeding '
                      'anyways.')

    if pd.isnull(data).any().any():
        raise ValueError('Data set contains NaNs.')

    # FIXME remove this check once feature factory is debugged.
    if (data == -99999).any().any():
        raise ValueError('Data set contains -99999s.')
