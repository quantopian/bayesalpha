import theano.tensor as tt
import pymc3 as pm
import numpy as np
from scipy import sparse, interpolate
from pymc3.distributions.distribution import draw_values


class NormalNonZero(pm.Normal):
    def logp(self, values):
        all_logp = pm.Normal.dist(mu=self.mu, sd=self.sd).logp(values)
        return tt.switch(tt.eq(values, 0), 0., all_logp)


class ScaledMvNormal(pm.Continuous):
    def __init__(self, mu, chol, sd_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mu = tt.as_tensor_variable(mu)
        self._chol = tt.as_tensor_variable(chol)
        self._sd_scale = tt.as_tensor_variable(sd_scale)
        if not chol.ndim == 2:
            raise ValueError('chol must be two-dimensional.')

    def logp(self, value):
        mu, chol, sd_scale = self._mu, self._chol, self._sd_scale
        if value.ndim not in [1, 2]:
            raise ValueError('Invalid value shape. Must be two-dimensional.')
        if value.ndim == 1:
            value = value.reshape((-1, value.shape[0]))

        if sd_scale.ndim == 2:
            trafo = (value - mu) / sd_scale
        elif sd_scale.ndim == 1:
            trafo = (value - mu) / sd_scale[:, None]
        else:
            assert False

        ok = ~tt.any(tt.isinf(trafo) | tt.isnan(trafo))
        trafo = tt.switch(ok, trafo, 1)

        logp = pm.MvNormal.dist(mu=tt.zeros([chol.shape[0]]),
                                chol=chol).logp(trafo)
        if sd_scale.ndim == 1:
            detfix = -chol.shape[0] * tt.log(sd_scale)
        else:
            detfix = -tt.log(sd_scale).sum(axis=-1)
        return tt.switch(ok, logp + detfix, -np.inf)

    def random(self, point=None, size=None):
        mu, chol, sd_scale = draw_values([self._mu, self._chol, self._sd_scale],
                                         point=point)
        normal = pm.Normal.dist(mu=mu, chol=chol).random(point, size)
        return normal * sd_scale[:, None]


class GPExponential(pm.Continuous):
    def __init__(self, mu, alpha, sigma, *args, **kwargs):
        self._mu = tt.as_tensor_variable(mu)
        self._alpha = tt.as_tensor_variable(alpha)
        self._sigma = tt.as_tensor_variable(sigma)
        self.mean = self.median = self.mode = mu
        super().__init__(*args, **kwargs)

    def logp(self, value):
        mu, alpha, sigma = self._mu, self._alpha, self._sigma
        value = value.reshape((-1, value.shape[-1]))
        k, n = value.shape  # TODO other shapes!

        delta = (value - mu) / sigma[..., None]

        corr = tt.exp(-alpha)
        mdiag_tau = - corr / (1 - corr ** 2)
        # diag_tau_middle = 1 - 2 * corr * mdiag_tau
        diag_tau_first = 1 - corr * mdiag_tau

        # Compute the cholesky decomposition of tau
        diag_chol = tt.sqrt(diag_tau_first)
        mdiag_chol = mdiag_tau / diag_chol

        if sigma.ndim == 1:
            logdet = 2 * k * n * np.log(diag_chol) / sigma
        else:
            logdet = 2 * n * (np.log(diag_chol) / sigma).sum()
        delta_trans = diag_chol * delta
        delta_trans = tt.set_subtensor(
            delta_trans[:, 1:], delta_trans[:, 1:] + mdiag_chol * delta[:, :-1])

        return -0.5 * (logdet + (delta_trans ** 2).sum())


def bspline_basis(n, eval_points, degree=3):
    n_knots = n + degree + 1
    knots = np.linspace(0, 1, n_knots - 2 * degree)
    knots = np.r_[[0] * degree, knots, [1] * degree]
    basis_funcs = interpolate.BSpline(knots, np.eye(n), k=degree)
    Bx = basis_funcs(eval_points)
    return sparse.csr_matrix(Bx)
