#!/usr/bin/env python
""" functions for input verification
"""

from typing import Optional, Union

import numpy as np
from _warnings import warn

from minterpy.global_settings import DEBUG, DEFAULT_DOMAIN, FLOAT_DTYPE, INT_DTYPE

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


    Parameters
    ----------
    domain : array_like or None
        Either one-dimensional domain (min,max), a stack of domains for each
        domain with shape (spatial_dimension,2).

    Returns
    -------
    verified_domain : array_like
        Stack of domains for each dimension with shape (spatial_dimension,2).

    Raises
    ------
    ValueError
        when no domain with the expected shape can be constructed from the input.

    """
    if domain is None:
        domain = np.repeat([DEFAULT_DOMAIN], spatial_dimension, axis=0)
    domain = np.require(domain, dtype=FLOAT_DTYPE)
    if domain.ndim == 1:
        domain = np.repeat([domain], spatial_dimension, axis=0)
    check_shape(domain, shape=(spatial_dimension, 2))
    return domain


def rectify_query_points(x, m):
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
    # TODO
    # convert into the expected shape
    out = results_placeholder.squeeze()
    if out.size == 1:  # do not return 0D but rather 1D array:
        out = out.reshape(1)
    return out


def check_type(a, expected_type=np.ndarray, *args, **kwargs):
    if not issubclass(type(a), expected_type):
        raise TypeError(
            f"input must be given as {expected_type} (encountered {type(a)})"
        )


def check_dtype(a: np.ndarray, expected_dtype):
    if a.dtype != expected_dtype:
        raise TypeError(
            f"input must be given as {expected_dtype} (encountered {a.dtype})"
        )


def check_values(a: np.ndarray, *args, **kwargs):
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        raise ValueError("values must not be NaN or infinity!")


def check_type_n_values(a: np.ndarray, *args, **kwargs):
    check_type(a, *args, **kwargs)
    check_values(a, *args, **kwargs)


def check_shape(
    a: np.ndarray, shape: Union[list, tuple] = None, dimensionality: int = None
):
    """
    example : check_shape(input_array, [8, 3, None, None])
    :param a: array to be checked
    :param shape: the expected shape.
        NOTE: non integer values will be interpreted as variable size in the respective dimension
    :param dimensionality:
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
    if size is None:
        size = a.shape[0]
    check_shape(a, shape=(size, size))


DOMAIN_WARN_MSG2 = "the grid points must fit the interpolation domain [-1;1]^m."
DOMAIN_WARN_MSG = (
    "this may lead to unexpected behaviour, "
    "e.g. rank deficiencies in the regression matrices, etc. ."
)


def check_domain_fit(points: np.ndarray):
    """checks weather a given array of points is properly formatted and spans the standard domain [-1,1]^m

    :param points: ndarray of shape (m, k) with m being the dimensionality and k the amount of points
    :raises ValueError or TypeError when any of the criteria are not satisfied
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
