import numpy as np
import pandas as pd
import pymc3 as pm
import bayesalpha
import pytest


def test_fit_author_model(data):
    trace = bayesalpha.fit_author_model(data,
                                        sampler_type='mcmc',
                                        sampler_args={
                                            'draws': 1,
                                            'tune': 1,
                                            'chains': 1
                                            }
                                        )
