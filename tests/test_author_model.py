import os
import pandas as pd
import bayesalpha
import pytest


@pytest.fixture
def data():
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                    os.path.dirname(__file__)))
    return pd.read_csv(__location__ + 'test_data/author_model_test_data.csv',
                       index_col=0)


def test_fit_author_model(data):
    trace = bayesalpha.fit_author_model(data,
                                        sampler_type='mcmc',
                                        sampler_args={
                                            'draws': 1,
                                            'tune': 1,
                                            'chains': 1
                                            }
                                        )
