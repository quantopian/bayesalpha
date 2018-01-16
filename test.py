import bayesalpha
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import pytest


@pytest.fixture(
    'module',
    [
      #  'diag',
        'dense',
        #'time-varying'
    ])
def Sigma_type(request):
    return request.param


@pytest.fixture('module', [1000])
def T(request):
    return request.param


@pytest.fixture('module')
def date_range(T):
    return pd.date_range('01-01-2000', periods=T, freq='B')


@pytest.fixture
def Sigma(Sigma_type, T):
    if Sigma_type == 'diag':
        return np.matrix([[0.000246, 0.], [0., 0.000093]], 'float32')
    elif Sigma_type == 'dense':
        return np.matrix([[0.000246, 0.000048], [0.000048, 0.000093]], 'float32')
    else:
        raise KeyError(Sigma_type)


@pytest.fixture('module')
def algo_gain():
    return np.array([0.0001, 0.0004], 'float32')


@pytest.fixture
def observations(date_range, T,
                 algo_gain, Sigma):
    if Sigma.shape[0] == T:
        t = ()
    else:
        t = T
    r = scipy.stats.multivariate_normal.rvs(algo_gain, Sigma, size=t, random_state=42)
    return pd.DataFrame(r, date_range, columns=['1', '2'])


@pytest.fixture
def algo_meta(date_range, T):
    mid = date_range[T//2]
    return pd.DataFrame(
        {'created_at': [mid, mid]}, index=['1', '2']
    )


def test_fit_population(observations, algo_meta, Sigma_type):
    trace = bayesalpha.fit_population(
        observations, algo_meta, sampler_args={'draws': 10, 'tune': 0, 'chains': 1},
        corr_type=Sigma_type
    )
