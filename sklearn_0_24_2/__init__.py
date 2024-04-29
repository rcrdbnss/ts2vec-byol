import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.extmath import (_incremental_mean_and_var,
                                   )
from sklearn.utils.sparsefuncs import (inplace_column_scale,
                                       mean_variance_axis, incr_mean_variance_axis)
from sklearn.utils.validation import (check_is_fitted, _check_sample_weight,
                                      FLOAT_DTYPES, _deprecate_positional_args)


# Use at least float64 for the accumulating functions to avoid precision issue
# see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
# as it is in case the float overflows
def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum.
    x : ndarray
        A numpy array to apply the accumulator function.
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x.
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function.

    Returns
    -------
    result
        The output of the accumulator function passed to this function.
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def _incremental_weighted_mean_and_var(X, sample_weight,
                                       last_mean,
                                       last_variance,
                                       last_weight_sum):
    """Calculate weighted mean and weighted variance incremental update.

    .. versionadded:: 0.24

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to use for mean and variance update.

    sample_weight : array-like of shape (n_samples,) or None
        Sample weights. If None, then samples are equally weighted.

    last_mean : array-like of shape (n_features,)
        Mean before the incremental update.

    last_variance : array-like of shape (n_features,) or None
        Variance before the incremental update.
        If None, variance update is not computed (in case scaling is not
        required).

    last_weight_sum : array-like of shape (n_features,)
        Sum of weights before the incremental update.

    Returns
    -------
    updated_mean : array of shape (n_features,)

    updated_variance : array of shape (n_features,) or None
        If None, only mean is computed.

    updated_weight_sum : array of shape (n_features,)

    Notes
    -----
    NaNs in `X` are ignored.

    `last_mean` and `last_variance` are statistics computed at the last step
    by the function. Both must be initialized to 0.0.
    The mean is always required (`last_mean`) and returned (`updated_mean`),
    whereas the variance can be None (`last_variance` and `updated_variance`).

    For further details on the algorithm to perform the computation in a
    numerically stable way, see [Finch2009]_, Sections 4 and 5.

    References
    ----------
    .. [Finch2009] `Tony Finch,
       "Incremental calculation of weighted mean and variance",
       University of Cambridge Computing Service, February 2009.
       <https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf>`_

    """
    # last = stats before the increment
    # new = the current increment
    # updated = the aggregated stats
    if sample_weight is None:
        return _incremental_mean_and_var(X, last_mean, last_variance,
                                         last_weight_sum)
    nan_mask = np.isnan(X)
    sample_weight_T = np.reshape(sample_weight, (1, -1))
    # new_weight_sum with shape (n_features,)
    new_weight_sum = np.dot(sample_weight_T,
                            ~nan_mask).ravel().astype(np.float64)
    total_weight_sum = _safe_accumulator_op(np.sum, sample_weight, axis=0)

    X_0 = np.where(nan_mask, 0, X)
    new_mean = np.average(X_0,
                          weights=sample_weight, axis=0).astype(np.float64)
    new_mean *= total_weight_sum / new_weight_sum
    updated_weight_sum = last_weight_sum + new_weight_sum
    updated_mean = (
            (last_weight_sum * last_mean + new_weight_sum * new_mean)
            / updated_weight_sum)

    if last_variance is None:
        updated_variance = None
    else:
        X_0 = np.where(nan_mask, 0, (X - new_mean) ** 2)
        new_variance = \
            _safe_accumulator_op(
                np.average, X_0, weights=sample_weight, axis=0)
        new_variance *= total_weight_sum / new_weight_sum
        new_term = (
                new_weight_sum *
                (new_variance +
                 (new_mean - updated_mean) ** 2))
        last_term = (
                last_weight_sum *
                (last_variance +
                 (last_mean - updated_mean) ** 2))
        updated_variance = (new_term + last_term) / updated_weight_sum

    return updated_mean, updated_variance, updated_weight_sum


def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


class StandardScaler_(TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve zero mean and unit
        variance. Generally this is calculated using `np.sqrt(var_)`. If a
        variance is zero, we can't achieve unit variance, and the data is left
        as-is, giving a scaling factor of 1. `scale_` is equal to `None`
        when `with_std=False`.

        .. versionadded:: 0.17
           *scale_*

    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False``.

    var_ : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_std=False``.

    n_samples_seen_ : int or ndarray of shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are no missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array of dtype int. If
        `sample_weights` are used it will be a float (if no missing data)
        or an array of dtype float that sums the weights seen so far.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler()
    >>> print(scaler.mean_)
    [0.5 0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]))
    [[3. 3.]]

    See Also
    --------
    scale : Equivalent function without the estimator API.

    :class:`~sklearn.decomposition.PCA` : Further removes the linear
        correlation across features with 'whiten=True'.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """  # noqa

    @_deprecate_positional_args
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.

        Returns
        -------
        self : object
            Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Online computation of mean and std on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        first_call = not hasattr(self, "n_samples_seen_")
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan', reset=first_call)
        n_features = X.shape[1]

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        # if n_samples_seen_ is an integer (i.e. no missing values), we need to
        # transform it to a NumPy array of shape (n_features,) required by
        # incr_mean_variance_axis and _incremental_variance_axis
        dtype = np.int64 if sample_weight is None else X.dtype
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = np.zeros(n_features, dtype=dtype)
        elif np.size(self.n_samples_seen_) == 1:
            self.n_samples_seen_ = np.repeat(
                self.n_samples_seen_, X.shape[1])
            self.n_samples_seen_ = \
                self.n_samples_seen_.astype(dtype, copy=False)

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            sparse_constructor = (sparse.csr_matrix
                                  if X.format == 'csr' else sparse.csc_matrix)

            if self.with_std:
                # First pass
                if not hasattr(self, 'scale_'):
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        mean_variance_axis(X, axis=0, weights=sample_weight,
                                           return_sum_weights=True)
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_,
                                                weights=sample_weight)
                # We force the mean and variance to float64 for large arrays
                # See https://github.com/scikit-learn/scikit-learn/pull/12338
                self.mean_ = self.mean_.astype(np.float64, copy=False)
                self.var_ = self.var_.astype(np.float64, copy=False)
            else:
                self.mean_ = None  # as with_mean must be False for sparse
                self.var_ = None
                weights = _check_sample_weight(sample_weight, X)
                sum_weights_nan = weights @ sparse_constructor(
                    (np.isnan(X.data), X.indices, X.indptr),
                    shape=X.shape)
                self.n_samples_seen_ += (
                    (np.sum(weights) - sum_weights_nan).astype(dtype)
                )
        else:
            # First pass
            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                if self.with_std:
                    self.var_ = .0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_std:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - np.isnan(X).sum(axis=0)

            elif sample_weight is not None:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    _incremental_weighted_mean_and_var(X, sample_weight,
                                                       self.mean_,
                                                       self.var_,
                                                       self.n_samples_seen_)
            else:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    _incremental_mean_and_var(X, self.mean_, self.var_,
                                              self.n_samples_seen_)

        # for backward-compatibility, reduce n_samples_seen_ to an integer
        # if the number of samples is the same for each feature (i.e. no
        # missing values)
        if np.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=False,
                                accept_sparse='csr', copy=copy,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X

    def _more_tags(self):
        return {'allow_nan': True,
                'preserves_dtype': [np.float64, np.float32]}
