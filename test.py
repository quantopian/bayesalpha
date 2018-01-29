import pymc3 as pm
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import bayesalpha
import bayesalpha.dists
import pytest


@pytest.fixture(
    'module',
    [
        'diag',
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


def test_fit_population_vi(observations, algo_meta, Sigma_type):
    trace = bayesalpha.fit_population(
        observations, algo_meta, sampler_type='vi',
        sampler_args={'n': 1},
        corr_type=Sigma_type
    )


def test_scaled_mv_normal_logp():
    cov = np.array([[0.246, 0.048], [0.048, 0.93]], 'float32')
    mean = np.arange(2)
    obs = np.random.rand(10, 2)
    scale = np.arange(20).reshape(10, 2)
    obs1 = obs * scale + mean[None, :]
    with pm.Model() as model1:
        for i in range(10):
            pm.MvNormal('mv%d' % i, mu=mean, cov=cov * scale[i][None, :] * scale[i][:, None], observed=obs1[i])

    with pm.Model() as model2:
        bayesalpha.dists.ScaledSdMvNormalNonZero('mv', mu=mean, cov=cov, scale_sd=scale, observed=obs1)
    logp1 = model1.logp({})
    logp2 = model2.logp({})
    np.testing.assert_allclose(logp1, logp2)