#!/usr/bin/env python
""" functions for input verification
"""

from typing import Optional, Union

import numpy as np
from _warnings import warn

from minterpy.global_settings import (DEBUG, DEFAULT_DOMAIN, FLOAT_DTYPE,
                                      INT_DTYPE)

__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"


def verify_domain(domain, spatial_dimension):
    """Building and verification of domains.

    This function builds a suitable domain as the cartesian product of a one-
    dimensional domain, or verifies the domain shape, of a multivariate domain is
    passed. If None is passed, the default domain is build from [-1,1].

    :param domain: Either one-dimensional domain ``(min,max)``, or a stack of domains for each domain with shape ``(spatial_dimension,2)``. If :class:`None` is passed, the ``DEFAULT_DOMAIN`` is repeated for each spatial dimentsion.
    :type domain: array_like, None
    :param spatial_dimension: Dimentsion of the domain space.
    :type spatial_dimension: int

    :return verified_domain: Stack of domains for each dimension with shape ``(spatial_dimension,2)``.
    :rtype: np.ndarray
    :raise ValueError: If no domain with the expected shape can be constructed from the input.

    """
    if domain is None:
        domain = np.repeat([DEFAULT_DOMAIN], spatial_dimension, axis=0)
    domain = np.require(domain, dtype=FLOAT_DTYPE)
    if domain.ndim == 1:
        domain = np.repeat([domain], spatial_dimension, axis=0)
    check_shape(domain, shape=(spatial_dimension, 2))
    return domain


def rectify_query_points(x, m):
    """Rectify input arguments.

    This function checks if a given input has the correct shape, or if the correct shape can be infered. For the latter it returns a version of the input with the correct shape. Correct shape means here ``(N,m)``, where ``N`` is the number of points and ``m`` the dimentsion of the domain space.

    :param x: Array of arguemnts passed to a function.
    :type x: np.ndarray

    :param m: Dimension of the domain space.
    :type m: int

    :raise ValueError: If the input array has not the expected dimensionality.

    :return: ``(nr_points,x)`` where ``nr_points`` is the number of points and ``x`` is a version of the input array with the correct shape. If ``x`` already had the right shape, it is passed without copying.
    :rtype: tuple(int,np.ndarray)

    """
    # TODO simplify. always require an unambiguous input shape!
    query_point_shape = x.shape
    if x.ndim == 1:
        if m == 1:  # -> every entry is a query point
            nr_points = query_point_shape[0]
        else:  # m > 1, interpret input as a single point -> dimensions must match!
            if len(x) != m:
                raise ValueError(
                    f"points x given as vector of shape {query_point_shape} (1D). "
                    f"detected dimensionality of the exponents however is {m}"
                )
            nr_points = 1
        x = x.reshape(nr_points, m)  # reshape to 2D
    else:
        nr_points, m_points = x.shape
        if m != m_points:
            raise ValueError(
                f"the dimensionality of the input points {m_points} "
                f"does not match the polynomial dimensionality {m}"
            )
    return nr_points, x


def rectify_eval_input(x, coefficients, exponents, verify_input):
    """Rectify input for evaluation.

    .. todo::
        - refac this based on the respective datatypes, e.g. :class:`MultiIndex` etc.
        - function signature if somewhat misleading.

    :param x:
    :type x: np.ndarray
    :param coefficients: The coefficients of the Newton polynomials. Note, format fixed such that 'lagrange2newton' conversion matrices can be passed as the Newton coefficients of all Lagrange monomials of a polynomial without prior transponation
    :type coefficients: np.ndarray
    :param exponents: a multi index ``alpha`` for every Newton polynomial corresponding to the exponents of this ``monomial``
    :type exponents: np.ndarray
    :param verify_input: weather the data types of the input should be checked.
    :type verify_input: bool

    :raise ValueError: If the number of coefficients does not match the number of monomials.
    :raise TypeError: if the input hasn't the expected dtype.

    :return: ``( N, coefficients, m, nr_points, nr_polynomials, x)`` with the rectifyed versions of the ``coefficients`` and the input array ``x``. Furthermore, return the number of exponents ``N``, the dimentsion of the domain space ``m``, the number of passed points ``nr_points`` and the number of polynomials ``nr_polynomials``.
    :rtype: (int, np.ndarray, int, int, int, np.ndarray)
    """
    N, m = exponents.shape
    if N == 0:
        raise ValueError("at least 1 monomial must be given")
    # NOTE: silently reshaping the input is dangerous,
    # because the value order could be changed without being noticed by the user -> unexpected results!
    nr_points, x = rectify_query_points(x, m)
    coeff_shape = coefficients.shape
    if len(coeff_shape) == 1:  # 1D: a single query polynomial
        nr_polynomials = 1
        coefficients = coefficients.reshape(N, nr_polynomials)  # reshape to 2D
    else:
        N_coeffs, nr_polynomials = coefficients.shape
        if N != N_coeffs:
            raise ValueError(
                f"the coefficient amount {N_coeffs} does not match the amount of monomials {N}"
            )

    if verify_input:
        expected_types = {
            "x": (x, FLOAT_DTYPE),
            "coefficients": (coefficients, FLOAT_DTYPE),
            "exponents": (exponents, INT_DTYPE),
        }
        for name, (value, exp_type) in expected_types.items():
            if value.dtype != exp_type:
                raise TypeError(
                    f"expected dtype {exp_type} for {name} but the dtype is {value.dtype}"
                )

    return N, coefficients, m, nr_points, nr_polynomials, x


def convert_eval_output(results_placeholder):
    """Converts an array to its squeezed version.

    The input array is copyed if necessary and the result is at least 1D.

    :param results_placeholder: Array to be converted.
    :type results_placeholder: np.ndarray

    :return: A (at least 1D) squeezed version of the input array.
    :rtype: np.ndarray

    .. todo::
        - use ``np.atleast_1D`` instead of the size check.


    """
    # TODO
    # convert into the expected shape
    out = results_placeholder.squeeze()
    if out.size == 1:  # do not return 0D but rather 1D array:
        out = out.reshape(1)
    return out


def check_type(a, expected_type=np.ndarray, *args, **kwargs):
    """Verify if the input has the expected type.

    :param a: Object to be checked.
    :type a: Any

    :param expected_type: The type which is check against. Default is ``np.ndarray``
    :type expected_type: type, optional

    :raise TypeError: if input object hasn't the expected type.

    .. todo::
        - why not use ``isinstance``?
        - why pass ``*args, **kwargs``?

    """
    if not issubclass(type(a), expected_type):
        raise TypeError(
            f"input must be given as {expected_type} (encountered {type(a)})"
        )


def check_dtype(a: np.ndarray, expected_dtype):
    """Verify if the input array has the expected dtype.

    :param a: Array to be checked.
    :type a: np.ndarray

    :param expected_dtype: The dtype which is check against.
    :type expected_dtype: type

    :raise TypeError: if input array hasn't the expected dtype.

    .. todo::
        - use ``is not`` instead of ``!=``.

    """
    if a.dtype != expected_dtype:
        raise TypeError(
            f"input must be given as {expected_dtype} (encountered {a.dtype})"
        )


def check_values(a: np.ndarray, *args, **kwargs):
    """Verify that the input array has neither ``NaN`` nor ``inf`` values.

    :param a: Array to be checked.
    :type a: np.ndarray

    :raise ValueError: if input array contains either ``NaN`` or ``inf`` values (or both).

    .. todo::
        - why pass ``*args, **kwargs``?

    """
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        raise ValueError("values must not be NaN or infinity!")


def check_type_n_values(a: np.ndarray, *args, **kwargs):
    """Verify that the input array has correct type and does neither contain ``NaN`` nor ``inf`` values.

    :param a: Array to be checked.
    :type a: np.ndarray

    .. todo::
        - why pass ``*args, **kwargs``?

    See Also
    --------
    check_type : verification of the type
    check_values : verification of the values

    """
    check_type(a, *args, **kwargs)
    check_values(a, *args, **kwargs)


def check_shape(
    a: np.ndarray, shape: Union[list, tuple] = None, dimensionality: int = None
):
    """Verify the shape of an input array.


    :param a: array to be checked.
    :type a: np.ndarray
    :param shape: the expected shape.Note, non integer values will be interpreted as variable size in the respective dimension. Default is :class:`None`.
    :type shape: {None,tuple,list}
    :param dimensionality: dimension of the domain space (right?). Default is :class:`None`
    :type dimensionality: {None,int}

    :raise ValueError: If input array hasn't the expected dimensionality.
    :raise ValueError: If input array hasn't the expected size.

    """
    if dimensionality is None:  # check for the dimensionality by the given shape:
        dimensionality = len(shape)
    if a.ndim != dimensionality:
        raise ValueError(
            f"expected {dimensionality}D array, but encountered array of dimensionality {len(a.shape)}"
        )
    if shape is None:
        return
    for dim, (true_size, expected_size) in enumerate(zip(a.shape, shape)):
        if isinstance(expected_size, int) and true_size != expected_size:
            raise ValueError(
                f"expected array of size {expected_size} in dimension {dim},"
                f" but encountered size {true_size}"
            )


def check_is_square(a: np.ndarray, size: Optional[int] = None):
    """Check if input array represents a square matrix.

    This is a special case of ``check_shape`` for ``shape = (size,size)``.

    :param a: Array to be checked
    :type a: np.ndarray

    :param size: expected length of each axes of the matrix. If :class:`None` is passed, ``size`` is set to the length of the first axis of the input array.
    :type size: int

    See Also
    --------
    check_shape : general verification of array shapes.

    """
    if size is None:
        size = a.shape[0]
    check_shape(a, shape=(size, size))


DOMAIN_WARN_MSG2 = "the grid points must fit the interpolation domain [-1;1]^m."
DOMAIN_WARN_MSG = (
    "this may lead to unexpected behaviour, "
    "e.g. rank deficiencies in the regression matrices, etc. ."
)


def check_domain_fit(points: np.ndarray):
    """ Checks weather a given array of points is properly formatted and spans the standard domain :math:`[-1,1]^m`.

    .. todo::
        - maybe remove the warnings.
        - generalise to custom ``internal_domain``

    :param points: array to be checked. Here ``m`` is the dimenstion of the domain and ``k`` is the number of points.
    :type points: np.ndarray, shape = (m, k)
    :raises ValueError: if the grid points do not fit into the domain :math:`[-1;1]^m`.
    :raises ValueError: if less than one point is passed.

    """
    # check first if the sample points are valid
    check_type_n_values(points)
    # check weather the points lie outside of the domain
    sample_max = np.max(points, axis=1)
    if not np.allclose(np.maximum(sample_max, 1.0), 1.0):
        raise ValueError(DOMAIN_WARN_MSG2 + f"violated max: {sample_max}")
    sample_min = np.min(points, axis=1)
    if not np.allclose(np.minimum(sample_min, -1.0), -1.0):
        raise ValueError(DOMAIN_WARN_MSG2 + f"violated min: {sample_min}")
    check_shape(points, dimensionality=2)
    nr_of_points, m = points.shape
    if nr_of_points == 0:
        raise ValueError("at least one point must be given")
    if nr_of_points == 1:
        return  # one point cannot span the domain
    if DEBUG:
        # check weather the points span the hole domain
        max_grid_val = np.max(sample_max)
        if not np.isclose(max_grid_val, 1.0):
            warn(
                f"the highest encountered value in the given points is {max_grid_val}  (expected 1.0). "
                + DOMAIN_WARN_MSG
            )
        min_grid_val = np.min(sample_min)
        if not np.isclose(min_grid_val, -1.0):
            warn(
                f"the smallest encountered value in the given points is {min_grid_val} (expected -1.0). "
                + DOMAIN_WARN_MSG
            )
