import theano.tensor as tt
import theano
import theano.tensor.extra_ops
import pymc3 as pm
import numpy as np
from scipy import sparse, interpolate
from pymc3.distributions.distribution import draw_values
from pymc3.distributions.dist_math import bound


class NormalNonZero(pm.Normal):
    def logp(self, value):
        all_logp = super(NormalNonZero, self).logp(value)
        return tt.switch(tt.eq(value, 0), 0., all_logp)


class ScaledSdMvNormalNonZero(pm.MvNormal):
    def __init__(self, *args, **kwargs):
        self.scale_sd = kwargs.pop('scale_sd')
        assert not args
        self._mu = kwargs.pop('mu')
        if isinstance(self._mu, tt.Variable):
            kwargs['mu'] = tt.zeros_like(self._mu)
        else:
            kwargs['mu'] = np.zeros_like(self._mu)
        super(ScaledSdMvNormalNonZero, self).__init__(**kwargs)

    def logp(self, value):
        scale_sd = self.scale_sd
        if scale_sd.ndim == 1:
            detfix = -pm.floatX(self._mu.shape[-1]) * tt.log(scale_sd)
        else:
            detfix = -tt.log(scale_sd).sum(axis=-1)
        mu = self._mu
        if value.ndim not in [1, 2]:
            raise ValueError('Invalid value shape. Must be two-dimensional.')
        if value.ndim == 1:
            value = value.reshape((-1, value.shape[0]))
        if scale_sd.ndim == 2:
            trafo = (value - mu) / scale_sd
        elif scale_sd.ndim == 1:
            trafo = (value - mu) / scale_sd[None, :]
        else:
            assert False
        ok = ~tt.any(tt.isinf(trafo) | tt.isnan(trafo))
        trafo = tt.switch(ok, trafo, 0.)
        logp = super(ScaledSdMvNormalNonZero, self).logp(trafo) + detfix
        logp = tt.switch(tt.eq(value, 0).any(-1), 0., logp)
        return tt.switch(ok, logp, -np.inf)

    def random(self, point=None, size=None):
        r = super(ScaledSdMvNormalNonZero, self).random(point=point, size=size)
        scale_sd, mu = draw_values([self.scale_sd, self._mu], point=point)
        r *= scale_sd
        r += mu
        return r


class GPExponential(pm.Continuous):
    def __init__(self, mu, alpha, sigma, *args, **kwargs):
        self._mu = tt.as_tensor_variable(mu)
        self._alpha = tt.as_tensor_variable(alpha)
        self._sigma = tt.as_tensor_variable(sigma)
        self.mean = self.median = self.mode = mu
        super(GPExponential, self).__init__(*args, **kwargs)

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


# The following is adapted from theano.sparse.basic, to fix Theano/Theano#6522


class Dot(theano.gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()

    def __str__(self):
        return "Sparse" + self.__class__.__name__

    def infer_shape(self, node, shapes):
        xshp, yshp = shapes
        x, y = node.inputs
        if x.ndim == 2 and y.ndim == 2:
            return [(xshp[0], yshp[1])]
        if x.ndim == 1 and y.ndim == 2:
            return [(yshp[1],)]
        if x.ndim == 2 and y.ndim == 1:
            return [(xshp[0],)]
        if x.ndim == 1 and y.ndim == 1:
            return [()]
        raise NotImplementedError()

    def make_node(self, x, y):
        dtype_out = theano.scalar.upcast(x.dtype, y.dtype)

        # Sparse dot product should have at least one sparse variable
        # as input. If the other one is not sparse, it has to be converted
        # into a tensor.
        if isinstance(x, sparse.spmatrix):
            x = theano.sparse.as_sparse_variable(x)
        if isinstance(y, sparse.spmatrix):
            y = theano.sparse.as_sparse_variable(y)
        x_is_sparse_var = theano.sparse.basic._is_sparse_variable(x)
        y_is_sparse_var = theano.sparse.basic._is_sparse_variable(y)

        if not x_is_sparse_var and not y_is_sparse_var:
            raise TypeError(
                "Sparse dot product should have at least one "
                "sparse variable as inputs, but the inputs are "
                "%s (%s) and %s (%s)." % (x, x.type, y, y.type))

        if x_is_sparse_var:
            broadcast_x = (False,) * x.ndim
        else:
            x = tt.as_tensor_variable(x)
            broadcast_x = x.type.broadcastable
            assert y.format in ["csr", "csc"]
            if x.ndim not in (1, 2):
                raise TypeError(
                    'theano.sparse.Dot: input 0 (0-indexed) must have ndim of '
                    '1 or 2, %d given.' % x.ndim)

        if y_is_sparse_var:
            broadcast_y = (False,) * y.ndim
        else:
            y = tt.as_tensor_variable(y)
            broadcast_y = y.type.broadcastable
            assert x.format in ["csr", "csc"]
            if y.ndim not in (1, 2):
                raise TypeError(
                    'theano.sparse.Dot: input 1 (1-indexed) must have ndim of '
                    '1 or 2, %d given.' % y.ndim)

        if len(broadcast_y) == 2:
            broadcast_out = broadcast_x[:-1] + broadcast_y[1:]
        elif len(broadcast_y) == 1:
            broadcast_out = broadcast_x[:-1]
        return theano.gof.Apply(
            self, [x, y], [tt.tensor(dtype=dtype_out, broadcastable=broadcast_out)])

    def perform(self, node, inputs, out):
        x, y = inputs
        out = out[0]
        x_is_sparse = theano.sparse.basic._is_sparse(x)
        y_is_sparse = theano.sparse.basic._is_sparse(y)

        if not x_is_sparse and not y_is_sparse:
            raise TypeError(x)

        rval = x * y

        if x_is_sparse and y_is_sparse:
            rval = rval.toarray()

        out[0] = theano._asarray(rval, dtype=node.outputs[0].dtype)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert (theano.sparse.basic._is_sparse_variable(x)
                or theano.sparse.basic._is_sparse_variable(y))
        rval = []

        if theano.sparse.basic._is_dense_variable(y):
            rval.append(tt.dot(gz, y.T))
        else:
            rval.append(dot(gz, y.T))
        if theano.sparse.basic._is_dense_variable(x):
            rval.append(tt.dot(x.T, gz))
        else:
            rval.append(dot(x.T, gz))

        return rval
_dot = Dot()


def dot(x, y):
    """
    Operation for efficiently calculating the dot product when
    one or all operands is sparse. Supported format are CSC and CSR.
    The output of the operation is dense.

    Parameters
    ----------
    x
        Sparse or dense matrix variable.
    y
        Sparse or dense matrix variable.

    Returns
    -------
    The dot product `x`.`y` in a dense format.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    At least one of `x` or `y` must be a sparse matrix.

    When the operation has the form dot(csr_matrix, dense)
    the gradient of this operation can be performed inplace
    by UsmmCscDense. This leads to significant speed-ups.

    """

    if hasattr(x, 'getnnz'):
        x = theano.sparse.as_sparse_variable(x)
    if hasattr(y, 'getnnz'):
        y = theano.sparse.as_sparse_variable(y)

    x_is_sparse_variable = theano.sparse.basic._is_sparse_variable(x)
    y_is_sparse_variable = theano.sparse.basic._is_sparse_variable(y)

    if not x_is_sparse_variable and not y_is_sparse_variable:
        raise TypeError()

    return _dot(x, y)


class BatchedMatrixInverse(tt.Op):
    """Computes the inverse of a matrix :math:`A`.

    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.

    Notes
    -----
    When possible, the call to this op will be optimized to the call
    of ``solve``.

    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        assert x.dim == 3
        return tt.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = np.linalg.inv(x).astype(x.dtype)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^{-1}}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.

        """
        x, = inputs
        xi = self.__call__(x)
        gz, = g_outputs
        # TT.dot(gz.T,xi)
        gx = tt.batched_dot(xi, gz.transpose(0, 2, 1))
        gx = tt.batched_dot(gx, xi)
        gx = -gx.transpose(0, 2, 1)
        return [gx]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return

            .. math:: \frac{\partial X^{-1}}{\partial X}V,

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: X^{-1} \cdot V \cdot X^{-1}.

        """
        x, = inputs
        xi = self.__call__(x)
        ev, = eval_points

        if ev is None:
            return [None]

        r = tt.batched_dot(xi, ev)
        r = tt.batched_dot(r, xi)
        r = -r
        return [r]

    def infer_shape(self, node, shapes):
        return shapes


batched_matrix_inverse = BatchedMatrixInverse()


class EQCorrMvNormal(pm.Continuous):
    def __init__(self, mu, std, corr, clust, *args, **kwargs):
        super(EQCorrMvNormal, self).__init__(*args, **kwargs)
        self.mu, self.std, self.corr, self.clust = map(
            tt.as_tensor_variable, [mu, std, corr, clust]
        )

    def logp(self, x):
        # -1/2 (x-mu) @ Sigma^-1 @ (x-mu)^T - 1/2 log(2pi^k|Sigma|)
        # Sigma = diag(std) @ Corr @ diag(std)
        # Sigma^-1 = diag(std^-1) @ Corr^-1 @ diag(std^-1)
        # Corr is a block matrix of special form
        #           +----------+
        # Corr = [[ | 1, b1, b1|,  0,  0,  0,..., 0]
        #         [ |b1,  1, b1|,  0,  0,  0,..., 0]
        #         [ |b1, b1,  1|,  0,  0,  0,..., 0]
        #           +-----------+----------+
        #         [  0,  0,  0, | 1, b2, b2|,..., 0]
        #         [  0,  0,  0, |b2,  1, b2|,..., 0]
        #         [  0,  0,  0, |b2, b2,  1|,..., 0]
        #                       +----------+
        #         [            ...                 ]
        #         [  0,  0,  0,   0,  0,  0 ,..., 1]]
        #
        # Corr = [[B1,  0,  0, ...,  0]
        #         [ 0, B2,  0, ...,  0]
        #         [ 0,  0, B3, ...,  0]
        #         [        ...        ]
        #         [ 0,  0,  0, ..., Bk]]
        #
        # Corr^-1 = [[B1^-1,     0,      0, ...,     0]
        #            [    0, B2^-1,      0, ...,     0]
        #            [    0,     0,  B3^-1, ...,     0]
        #            [              ...               ]
        #            [    0,     0,      0, ..., Bk^-1]]
        #
        # |B| matrix of rank r is easy
        # https://math.stackexchange.com/a/1732839
        # Let D = eye(r) * (1-b)
        # Then B = D + b * ones((r, r))
        # |B| = (1-b) ** r + b * r * (1-b) ** (r-1)
        #
        # Inverse B^-1 is easy as well
        # https://math.stackexchange.com/a/1766118
        # let
        # c = 1/b + r*1/(1-b)
        # (B^-1)ii = 1/(1-b) - 1/(c*(1-b)**2)
        # (B^-1)ij =         - 1/(c*(1-b)**2)
        #
        # assuming
        # z = (x - mu) / std
        # we have det fix
        # detfix = sum(log(std))
        #
        # now we need to compute z @ Corr^-1 @ z^T
        # note that B can be unique per timestep
        # so we need z_t @ Corr_t^-1 @ z_t^T in perfect
        # z_t @ Corr_t^-1 @ z_t^T is a sum of block terms
        # z_ct @ B_ct^-1 @ z_ct^T = (B^-1)_iict * sum(z_ct**2) + (B^-1)_ijct*sum_{i!=j}(z_ict * z_jct)

        clust_ids, clust_pos, clust_counts = tt.extra_ops.Unique(return_inverse=True, return_counts=True)(self.clust)
        clust_order = tt.argsort(clust_pos)
        mu = self.mu
        corr = self.corr[..., clust_ids]
        std = self.std
        if std.ndim == 0:
            std = tt.repeat(std, x.shape[-1])
        if std.ndim == 1:
            std = std[None, :]
        if corr.ndim == 1:
            corr = corr[None, :]
        z = (x - mu)/std
        z = z[..., clust_order]
        detfix = -tt.log(std).sum(-1)
        # following the notation above
        r = clust_counts
        b = corr
        detB = (1.-b) ** r + b * r * (1.-b) ** (r-1)
        c = 1 / b + r / (1. - b)
        invBij = -1./(c*(1.-b)**2)
        invBii = 1./(1.-b) + invBij
        invBij = tt.repeat(invBij, clust_counts, axis=-1)
        invBii = tt.repeat(invBii, clust_counts, axis=-1)

        # to compute (Corr^-1)_ijt*sum_{i!=j}(z_it * z_jt) we use masked cross products
        mask = tt.arange(x.shape[-1])[None, :]
        mask = tt.repeat(mask, x.shape[-1], axis=0)
        mask = tt.maximum(mask, mask.T)
        block_end_pos = tt.cumsum(r)
        block_end_pos = tt.repeat(block_end_pos, clust_counts)
        mask = tt.lt(mask, block_end_pos)
        mask = tt.and_(mask, mask.T)
        mask = tt.fill_diagonal(mask.astype('float32'), 0.)  # type: tt.TensorVariable

        invBiizizi_sum = ((z**2) * invBii).sum(-1)
        invBijzizj_sum = (
            (z.dimshuffle(0, 1, 'x')
             * mask.dimshuffle('x', 0, 1)
             * z.dimshuffle(0, 'x', 1))
            * invBij.dimshuffle(0, 1, 'x')
        ).sum([-1, -2])
        quad = invBiizizi_sum + invBijzizj_sum
        k = pm.floatX(x.shape[-1])
        logp = (
            detfix
            - .5 * (
                quad
                + pm.floatX(np.log(np.pi*2)) * k
                + tt.log(detB).sum(-1)
            )
        )

        return bound(logp,
                     -1. < corr < 1.,
                     std > 0.,
                     broadcast_conditions=False)

    def random(self, point=None, size=None):
        mu, std, corr, clust = draw_values([self.mu, self.std, self.corr, self.clust], point=point)
        return self.st_random(mu, std, corr, clust, size=size)

    @staticmethod
    def st_random(mu, std, corr, clust, size=None):
        k = mu.shape[-1]
        if std.ndim == 0:
            std = tt.repeat(std, k)
        if std.ndim == 1:
            std = std[None, :]
        if corr.ndim == 1:
            corr = corr[None, :]
        clust_ids, clust_pos, clust_counts = np.unique(clust, return_inverse=True, return_counts=True)
        corr = corr[..., clust_ids]
        block_end_pos = np.cumsum(clust_counts)
        block_end_pos = np.repeat(block_end_pos, clust_counts)
        mask = np.arange(k)[None, :]
        mask = np.repeat(mask, k, axis=0)
        mask = np.maximum(mask, mask.T)
        mask = (mask < block_end_pos) & (mask < block_end_pos).T
        corr = np.repeat(corr, clust_counts, axis=-1)[..., None]
        corr = corr * mask[None, :]
        corr[:, np.arange(k), np.arange(k)] = 1
        std = std[..., None]
        cov = std * corr * std.swapaxes(-1, -2)
        chol = np.linalg.cholesky(cov)
        standard_normal = np.random.standard_normal(size)
        return mu + np.dot(standard_normal, chol.swapaxes(-1, -2))
