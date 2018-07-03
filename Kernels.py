"""
Modified Kernel file from scikit-learn.
Modification to the RBF-, Constant-Kernel, summation and multiplication:
    Add derivative with respect to the input parameters to the kernel matrix
    Derivative of the derived kernel matrix for the hyper parameter optimization
    change in function calling


Kernels for Gaussian process regression and classification.
The kernels in this module allow kernel-engineering, i.e., they can be
combined via the "+" and "*" operators or be exponentiated with a scalar
via "**". These sum and product expressions can also contain scalar values,
which are automatically converted to a constant kernel.

All kernels allow (analytic) gradient-based hyperparameter optimization.
The space of hyperparameters can be specified by giving lower und upper
boundaries for the value of each hyperparameter (the search space is thus
rectangular). Instead of specifying bounds, hyperparameters can also be
declared to be "fixed", which causes these hyperparameters to be excluded from
optimization.
"""

import scipy.spatial.distance as _spdist
from abc import ABCMeta, abstractmethod
import sklearn.gaussian_process
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.externals import six
from sklearn.base import clone
from sklearn.externals.funcsigs import signature

# TODO Exponentiation Class  derivative with respect to x, y
# TODO Matern Class  derivative with respect to x, y
# ToDo modify description
# Todo diag function for RBF-, Constant-Kernel


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError("Anisotropic kernel must have the same number of "
                         "dimensions as data (%d!=%d)"
                         % (length_scale.shape[0], X.shape[1]))
    return length_scale


class Hyperparameter(namedtuple('Hyperparameter',('name', 'value_type', 'bounds','n_elements', 'fixed'))):
    """A kernel hyperparameter's specification in form of a namedtuple.

    .. versionadded:: 0.18

    Attributes
    ----------
    name : string
        The name of the hyperparameter. Note that a kernel using a
        hyperparameter with name "x" must have the attributes self.x and
        self.x_bounds

    value_type : string
        The type of the hyperparameter. Currently, only "numeric"
        hyperparameters are supported.

    bounds : pair of floats >= 0 or "fixed"
        The lower and upper bound on the parameter. If n_elements>1, a pair
        of 1d array with n_elements each may be given alternatively. If
        the string "fixed" is passed as bounds, the hyperparameter's value
        cannot be changed.

    n_elements : int, default=1
        The number of elements of the hyperparameter value. Defaults to 1,
        which corresponds to a scalar hyperparameter. n_elements > 1
        corresponds to a hyperparameter which is vector-valued,
        such as, e.g., anisotropic length-scales.

    fixed : bool, default: None
        Whether the value of this hyperparameter is fixed, i.e., cannot be
        changed during hyperparameter tuning. If None is passed, the "fixed" is
        derived based on the given bounds.

    """
    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __init__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __new__(cls, name, value_type, bounds, n_elements=1, fixed=None):
        if not isinstance(bounds, six.string_types) or bounds != "fixed":
            bounds = np.atleast_2d(bounds)
            if n_elements > 1:  # vector-valued parameter
                if bounds.shape[0] == 1:
                    bounds = np.repeat(bounds, n_elements, 0)
                elif bounds.shape[0] != n_elements:
                    raise ValueError("Bounds on %s should have either 1 or "
                                     "%d dimensions. Given are %d"
                                     % (name, n_elements, bounds.shape[0]))

        if fixed is None:
            fixed = isinstance(bounds, six.string_types) and bounds == "fixed"
        return super(Hyperparameter, cls).__new__(
            cls, name, value_type, bounds, n_elements, fixed)

    # This is mainly a testing utility to check that two hyperparameters
    # are equal.
    def __eq__(self, other):
        return (self.name == other.name and
                self.value_type == other.value_type and
                np.all(self.bounds == other.bounds) and
                self.n_elements == other.n_elements and
                self.fixed == other.fixed)


class Kernel(six.with_metaclass(ABCMeta)):
    """Base class for all kernels.

    .. versionadded:: 0.18
    """

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if (parameter.kind != parameter.VAR_KEYWORD and
                    parameter.name != 'self'):
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError("scikit-learn kernels should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    def set_params(self, **params):
        """Set the parameters of this kernel.

        The method works on simple kernels as well as on nested kernels.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for kernel %s. '
                                     'Check the list of available parameters '
                                     'with `kernel.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for kernel %s. '
                                     'Check the list of available parameters '
                                     'with `kernel.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def clone_with_theta(self, theta):
        """Returns a clone of self with given hyperparameters theta. """
        cloned = clone(self)
        cloned.theta = theta
        return cloned

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = []
        for attr in dir(self):
            if attr.startswith("hyperparameter_"):
                r.append(getattr(self, attr))
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                theta.append(params[hyperparameter.name])
        if len(theta) > 0:
            return np.log(np.hstack(theta))
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                params[hyperparameter.name] = np.exp(
                    theta[i:i + hyperparameter.n_elements])
                i += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = np.exp(theta[i])
                i += 1

        if i != len(theta):
            raise ValueError("theta has not the correct number of entries."
                             " Should be %d; given are %d"
                             % (i, len(theta)))
        self.set_params(**params)

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = []
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                bounds.append(hyperparameter.bounds)
        if len(bounds) > 0:
            return np.log(np.vstack(bounds))
        else:
            return np.array([])

    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        params_a = self.get_params()
        params_b = b.get_params()
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        return True

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map("{0:.3g}".format, self.theta)))

    @abstractmethod
    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Evaluate the kernel."""

    @abstractmethod
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """

    @abstractmethod
    def is_stationary(self):
        """Returns whether the kernel is stationary. """


class NormalizedKernelMixin(object):
    """Mixin for kernels which are normalized: k(X, X)=1.

    .. versionadded:: 0.18
    """

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.ones(X.shape[0])


class StationaryKernelMixin(object):
    """Mixin for kernels which are stationary: k(X, Y)= f(X-Y).

    .. versionadded:: 0.18
    """

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return True


class CompoundKernel(Kernel):
    """Kernel which is composed of a set of other kernels.

    .. versionadded:: 0.18
    """

    def __init__(self, kernels):
        self.kernels = kernels

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return dict(kernels=self.kernels)

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return np.hstack([kernel.theta for kernel in self.kernels])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        k_dims = self.k1.n_dims
        for i, kernel in enumerate(self.kernels):
            kernel.theta = theta[i * k_dims:(i + 1) * k_dims]

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return np.vstack([kernel.bounds for kernel in self.kernels])

    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Note that this compound kernel returns the results of all simple kernel
        stacked along an additional axis.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y, n_kernels)
            Kernel k(X, Y)

        K_gradient : array, shape (n_samples_X, n_samples_X, n_dims, n_kernels)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if eval_gradient:
            K = []
            K_grad = []
            for kernel in self.kernels:
                K_single, K_grad_single = kernel(X, Y, eval_gradient)
                K.append(K_single)
                K_grad.append(K_grad_single[..., np.newaxis])
            return np.dstack(K), np.concatenate(K_grad, 3)
        else:
            return np.dstack([kernel(X, Y, eval_gradient)
                              for kernel in self.kernels])

    def __eq__(self, b):
        if type(self) != type(b) or len(self.kernels) != len(b.kernels):
            return False
        return np.all([self.kernels[i] == b.kernels[i]
                       for i in range(len(self.kernels))])

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return np.all([kernel.is_stationary() for kernel in self.kernels])

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X, n_kernels)
            Diagonal of kernel k(X, X)
        """
        return np.vstack([kernel.diag(X) for kernel in self.kernels]).T


class KernelOperator(Kernel):
    """Base class for all kernel operators.

    .. versionadded:: 0.18
    """

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict(k1=self.k1, k2=self.k2)
        if deep:
            deep_items = self.k1.get_params().items()
            params.update(('k1__' + k, val) for k, val in deep_items)
            deep_items = self.k2.get_params().items()
            params.update(('k2__' + k, val) for k, val in deep_items)

        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.k1.hyperparameters:
            r.append(Hyperparameter("k1__" + hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.n_elements))
        for hyperparameter in self.k2.hyperparameters:
            r.append(Hyperparameter("k2__" + hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.n_elements))
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return np.append(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        k1_dims = self.k1.n_dims
        self.k1.theta = theta[:k1_dims]
        self.k2.theta = theta[k1_dims:]

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        if self.k1.bounds.size == 0:
            return self.k2.bounds
        if self.k2.bounds.size == 0:
            return self.k1.bounds
        return np.vstack((self.k1.bounds, self.k2.bounds))

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return (self.k1 == b.k1 and self.k2 == b.k2) \
            or (self.k1 == b.k2 and self.k2 == b.k1)

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return self.k1.is_stationary() and self.k2.is_stationary()


class Sum(KernelOperator):
    """Sum-kernel k1 + k2 of two kernels k1 and k2.

    The resulting kernel is defined as
    k_sum(X, Y) = k1(X, Y) + k2(X, Y)

    .. versionadded:: 0.18

    Parameters
    ----------
    k1 : Kernel object
        The first base-kernel of the sum-kernel

    k2 : Kernel object
        The second base-kernel of the sum-kernel

    """

    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """

        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, dx=dx, dy=dy, eval_gradient=True)
            K2, K2_gradient = self.k2(X, Y, dx=dx, dy=dy, eval_gradient=True)

            if dx != 0 or dy != 0:
                 if isinstance(self.k1, ConstantKernel):
                     K1_gradient = np.zeros_like(K1_gradient)
                     K1 = np.zeros_like(K1)

                 if isinstance(self.k2, ConstantKernel):
                    K2_gradient = np.zeros_like(K2_gradient)
                    K2 = np.zeros_like(K2)

            return K1 + K2, np.dstack((K1_gradient, K2_gradient))

        else:
                K1 = self.k1(X, Y, dx=dx, dy=dy)
                K2 = self.k2(X, Y, dx=dx, dy=dy)

                if dx != 0 or dy != 0:
                    if isinstance(self.k1, ConstantKernel):
                        K1 = np.zeros_like(K1)
                    if isinstance(self.k2, ConstantKernel):
                        K2 = np.zeros_like(K2)

        return K1 + K2 #self.k1(X, Y) + self.k2(X, Y)

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.k1.diag(X) + self.k2.diag(X)

    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)


class Product(KernelOperator):
    """Product-kernel k1 * k2 of two kernels k1 and k2.

    The resulting kernel is defined as
    k_prod(X, Y) = k1(X, Y) * k2(X, Y)

    .. versionadded:: 0.18

    Parameters
    ----------
    k1 : Kernel object
        The first base-kernel of the product-kernel

    k2 : Kernel object
        The second base-kernel of the product-kernel

    """

    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, dx=dx, dy=dy, eval_gradient=True)
            K2, K2_gradient = self.k2(X, Y, dx=dx, dy=dy, eval_gradient=True)

            if dx != 0 or dy != 0:
                if isinstance(self.k1, ConstantKernel):
                    K1_gradient = np.zeros_like(K1_gradient)
                    K1 = np.zeros_like(K1)
                if isinstance(self.k2, ConstantKernel):
                    K2_gradient = np.zeros_like(K2_gradient)
                    K2 = np.zeros_like(K2)

                K1_0, K1_gradient_0 = self.k1(X, Y, eval_gradient=True)
                K2_0, K2_gradient_0 = self.k2(X, Y, eval_gradient=True)

                return K1*K2_0 + K2 * K1_0, \
                       np.dstack((K1_gradient * K2_0[:, :, np.newaxis] + K1_gradient_0 * K2[:, :, np.newaxis],
                                  K2_gradient * K1_0[:, :, np.newaxis]+ K2_gradient_0 * K1[:, :, np.newaxis]))

            return K1 * K2, np.dstack((K1_gradient * K2[:, :, np.newaxis], K2_gradient * K1[:, :, np.newaxis]))
        else:
            K1 = self.k1(X, Y, dx=dx, dy=dy, eval_gradient=False)
            K2 = self.k2(X, Y, dx=dx, dy=dy, eval_gradient=False)
            if dx != 0 or dy != 0:
                if isinstance(self.k1, ConstantKernel):
                    K1 = np.zeros_like(K1)
                if isinstance(self.k2, ConstantKernel):
                    K2 = np.zeros_like(K2)
                return K1*self.k2(X, Y, dx=0, dy=0, eval_gradient=False) \
                       + K2 * self.k1(X, Y, dx=0, dy=0, eval_gradient=False)
            return K1 * K2

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.k1.diag(X) * self.k2.diag(X)

    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)


class Exponentiation(Kernel):
    """Exponentiate kernel by given exponent.

    The resulting kernel is defined as
    k_exp(X, Y) = k(X, Y) ** exponent

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : Kernel object
        The base kernel

    exponent : float
        The exponent for the base kernel

    """
    def __init__(self, kernel, exponent):
        self.kernel = kernel
        self.exponent = exponent

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict(kernel=self.kernel, exponent=self.exponent)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(('kernel__' + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(Hyperparameter("kernel__" + hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.n_elements))
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        self.kernel.theta = theta

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return (self.kernel == b.kernel and self.exponent == b.exponent)

    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if eval_gradient:
            K, K_gradient = self.kernel(X, Y, eval_gradient=True)
            K_gradient *= \
                self.exponent * K[:, :, np.newaxis] ** (self.exponent - 1)
            return K ** self.exponent, K_gradient
        else:
            K = self.kernel(X, Y, eval_gradient=False)
            return K ** self.exponent

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.kernel.diag(X) ** self.exponent

    def __repr__(self):
        return "{0} ** {1}".format(self.kernel, self.exponent)

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return self.kernel.is_stationary()


class ConstantKernel(StationaryKernelMixin, Kernel):
    """Constant kernel.

    Can be used as part of a product-kernel where it scales the magnitude of
    the other factor (kernel) or as part of a sum-kernel, where it modifies
    the mean of the Gaussian process.

    k(x_1, x_2) = constant_value for all x_1, x_2

    .. versionadded:: 0.18

    Parameters
    ----------
    constant_value : float, default: 1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on constant_value

    """
    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5)):
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds

    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter(
            "constant_value", "numeric", self.constant_value_bounds)

    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        # if Y is None:
        #     Y = X
        # elif eval_gradient:
        #     raise ValueError("Gradient can only be evaluated when Y is None.")

        K = self.constant_value * np.ones((X.shape[0], Y.shape[0]))
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (K, self.constant_value
                        * np.ones((X.shape[0], X.shape[0], 1)))
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.constant_value * np.ones(X.shape[0])

    def __repr__(self):
        return "{0:.3g}**2".format(np.sqrt(self.constant_value))


class RBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial-basis function kernel (aka squared-exponential kernel).

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length-scale
    parameter length_scale>0, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.

    .. versionadded:: 0.18

    Parameters
    -----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale

    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        dists = cdist(X / length_scale, Y / length_scale, metric='sqeuclidean')
        K = np.exp(-.5 * dists)

        if not self.anisotropic or length_scale.shape[0] == 1:
            if dx != 0 or dy != 0:
                if dx == dy:
                    K = K * (1 - np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])**2 / length_scale**2) / length_scale**2 # J_ii
                elif dx == 0: # and dy != 0:
                    K = K * np.subtract.outer(X[:, dy-1].T, Y[:, dy-1])/length_scale**2 # G
                elif dy == 0: # and dx != 0:
                    K = -K * np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])/length_scale**2 # K_prime
                else:
                    K = -K * np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])*np.subtract.outer(X[:, dy-1].T, Y[:, dy-1])\
                                 /length_scale**4 # J_ij
        else:
            if dx != 0 or dy != 0:
                if dx == dy:
                    K = K * (1 - np.subtract.outer(X[:,dx-1].T, Y[:, dx-1])**2/length_scale[dx-1]**2)\
                             /length_scale[dx-1]**2
                elif dx == 0:
                    K = K * np.subtract.outer(X[:, dy-1].T, Y[:, dy-1])/length_scale[dy-1]**2
                elif dy == 0:
                    K = -K * np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])/length_scale[dx-1]**2
                else:
                    K = - K * np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])*np.subtract.outer(X[:, dy-1].T, Y[:, dy-1])\
                             / (length_scale[dx-1]**2 * length_scale[dy-1]**2)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], Y.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                if dx == 0 and dy == 0:
                    K_gradient = (K * dists)[:, :, np.newaxis]
                elif dx != 0 and dy == 0:
                    K_gradient = (K * (dists - 2))[:, :, np.newaxis]
                elif dy != 0 and dx == 0:
                    K_gradient = (K * (dists - 2))[:, :, np.newaxis]
                    # K_gradient = (K * (dists/length_scale**2 - 2/length_scale**2))[:, :, np.newaxis]
                else:
                    if dx == dy:
                        K_gradient = (K * (dists - 4) + 2 * np.exp(-0.5*dists)/length_scale**2)[:, :, np.newaxis]
                        # K_gradient = (np.exp(-.5 * dists) / length_scale ** 2 * (5*dists - 2 - dists **2))[:, :, np.newaxis]
                        # K_gradient = (np.exp(-0.5*dists)*(dists**2/length_scale**3-1./length_scale**5-4*(dists-1/length_scale**3)-2/length_scale**3))[:, :, np.newaxis]
                    else:
                        K_gradient = (K * (dists - 4))[:, :, np.newaxis]
                        # K_gradient = (-np.exp(-.5 * dists) * dists/length_scale**2 * (4 - dists))[:, :, np.newaxis]
                        # K_gradient = (np.exp(-.5 * dists) * dists**2 / length_scale ** 3 * (dists**2 - 4))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:

                # We need to recompute the pairwise dimension-wise distances
                grad = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2 / (length_scale ** 2)
                if dx == 0 and dy == 0:
                    K_gradient = grad * K[..., np.newaxis]

                elif (dx != 0 and dy == 0):
                    K_gradient = grad * K[..., np.newaxis]
                    K_gradient[:, :, dx - 1] = K_gradient[:, :, dx - 1] - 2 * K
                elif (dy != 0 and dx == 0):
                    K_gradient = grad * K[..., np.newaxis]
                    K_gradient[:, :, dy-1] = K_gradient[:, :, dy-1] - 2*K
                else:
                    if dx == dy:
                        K_gradient = grad * K[..., np.newaxis]
                        K_gradient[:, :, dx - 1] = K_gradient[:, :, dx - 1] - np.exp(-0.5 * dists) * (2 /
                                            length_scale[dx-1] ** 2 - 4 * grad[:, :, dx - 1] / length_scale[dx-1] ** 2)

                    else:
                        K_gradient = grad * K[..., np.newaxis]
                        K_gradient[:, :, dx - 1] = K_gradient[:, :, dx - 1] - 2*K
                        K_gradient[:, :, dy - 1] = K_gradient[:, :, dy - 1] - 2*K
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])

