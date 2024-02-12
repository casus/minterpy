#!/usr/bin/env python
""" functions for input verification
"""

from typing import Optional, Sized, Tuple, TypeVar, Union

import numpy as np
from _warnings import warn

from minterpy.global_settings import DEBUG, DEFAULT_DOMAIN, FLOAT_DTYPE, INT_DTYPE


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

    This function checks if a given input has the correct shape, or if the correct shape can be inferred. For the latter it returns a version of the input with the correct shape. Correct shape means here ``(N,m)``, where ``N`` is the number of points and ``m`` the dimension of the domain space.

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


def check_dimensionality(xx: np.ndarray, dimensionality: int) -> None:
    """Verify the dimensionality of a given array.

    Use this verification function when its expected dimensionality is known.

    Parameters
    ----------
    xx : np.ndarray
        A given array to verify.
    dimensionality : int
        The expected dimensionality (i.e., the number of dimensions)
        of the array.

    Raises
    ------
    ValueError
        If the input array is not of the expected dimension.

    Examples
    --------
    >>> check_dimensionality(np.array([1, 2, 3]), dimensionality=1)
    >>> yy = np.array([
    ...     [1, 2, 3, 4],
    ...     [5, 6, 7, 8],
    ... ])
    >>> check_dimensionality(yy, dimensionality=2)
    >>> check_dimensionality(yy, dimensionality=1)  # Wrong dimensionality
    Traceback (most recent call last):
    ...
    ValueError: 1D array is expected; got instead 2D!
    """
    if xx.ndim != dimensionality:
        raise ValueError(
            f"{dimensionality}D array is expected; got instead {xx.ndim}D!"
        )


def check_shape(xx: np.ndarray, shape: Tuple[int, ...]):
    """Verify the shape of a given array.

    Use this verification function when its expected shape (given as a tuple)
    is known.

    Parameters
    ----------
    xx : np.ndarray
        A given array to verify.
    shape : Tuple[int, ...]
        The expected shape of the array.

    Raises
    ------
    ValueError
        If the input array is not of the expected shape.

    Examples
    --------
    >>> check_shape(np.array([1, 2, 3]), shape=(3, ))
    >>> yy = np.array([
    ...     [1, 2, 3, 4],
    ...     [5, 6, 7, 8],
    ... ])
    >>> check_shape(yy, shape=(2, 4))
    >>> check_shape(yy, shape=(1, 5))  # Wrong shape
    Traceback (most recent call last):
    ...
    ValueError: Array of shape (1, 5) is expected; got instead (2, 4)!
    >>> check_shape(yy, shape=(2, 4, 1))  # Wrong dimensionality
    Traceback (most recent call last):
    ...
    ValueError: 3D array is expected; got instead 2D!
    """
    # Check dimensionality
    check_dimensionality(xx, dimensionality=len(shape))

    # Check shape
    if xx.shape != shape:
        raise ValueError(
            f"Array of shape {shape} is expected; got instead {xx.shape}!"
        )


DOMAIN_WARN_MSG2 = "the grid points must fit the interpolation domain [-1;1]^m."
DOMAIN_WARN_MSG = (
    "this may lead to unexpected behaviour, "
    "e.g. rank deficiencies in the regression matrices, etc. ."
)


def check_domain_fit(points: np.ndarray):
    """Checks weather a given array of points is properly formatted and spans the standard domain :math:`[-1,1]^m`.

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
    check_dimensionality(points, dimensionality=2)
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


def check_values(xx: Union[int, float, np.ndarray], **kwargs):
    """Verify that the input array has neither ``NaN`` nor ``inf`` values.

    Parameters
    ----------
    xx : Union[int, float, :class:`numpy:numpy.ndarray`]
        The scalar or array to be checked.
    **kwargs
        Keyword arguments with Boolean as values, if ``True`` then the invalid
        value is allowed. The keys are ``nan`` (check for ``NaN`` values),
        ``inf`` (check for ``inf`` values), ``zero`` (check for 0 values),
        and ``negative`` (check for negative values).
        If any of those set to ``True``,
        the given value will raise an exception.
        The default is ``NaN`` and ``inf`` values are not allowed, while
        zero or negative values are allowed.

    Raises
    ------
    ValueError
        If the scalar value or the given array contains any of the specified
        invalid values(e.g., ``NaN``, ``inf``, zero, or negative).

    Examples
    --------
    >>> check_values(10)  # valid
    >>> check_values(np.nan)  # Default, no nan
    Traceback (most recent call last):
    ...
    ValueError: Invalid value(s) (NaN, inf, negative, zero)!
    >>> check_values(np.inf)  # Default, no inf
    Traceback (most recent call last):
    ...
    ValueError: Invalid value(s) (NaN, inf, negative, zero)!
    >>> check_values(np.zeros((3, 2)), zero=False)  # No zero value is allowed
    Traceback (most recent call last):
    ...
    ValueError: Invalid value(s) (NaN, inf, negative, zero)!
    >>> check_values(-10, negative=False)  # No negative value is allowed
    Traceback (most recent call last):
    ...
    ValueError: Invalid value(s) (NaN, inf, negative, zero)!
    """
    # Parse keyword arguments
    kwargs = dict((key.lower(), val) for key, val in kwargs.items())
    nan = kwargs.get("nan", False)
    inf = kwargs.get("inf", False)
    zero = kwargs.get("zero", True)
    negative = kwargs.get("negative", True)

    # Check values
    is_nan = False if nan else np.any(np.isnan(xx))
    is_inf = False if inf else np.any(np.isinf(xx))
    is_zero = False if zero else np.any(xx == 0)
    is_negative = False if negative else np.any(xx < 0)

    if is_nan or is_inf or is_zero or is_negative:
        raise ValueError(
            "Invalid value(s) (NaN, inf, negative, zero)!"
        )


def verify_spatial_dimension(spatial_dimension: int) -> int:
    """Verify if the value of a given spatial dimension is valid.

    Parameters
    ----------
    spatial_dimension : int
        Spatial dimension to verify; the value of a spatial dimension must be
        strictly positive (> 0). ``spatial_dimension`` may not necessarily be
        an `int` but it must be a single whole number.

    Returns
    -------
    int
        Verified spatial dimension. If the input is not an `int`,
        the function does a type conversion to an `int` if possible.

    Raises
    ------
    TypeError
        If ``spatial_dimension`` is not of a correct type, i.e., its
        strict-positiveness cannot be verified or the conversion to `int`
        cannot be carried out.
    ValueError
        If ``spatial_dimension`` is, for example, not a positive
        or a whole number.

    Examples
    --------
    >>> verify_spatial_dimension(2)  # int
    2
    >>> verify_spatial_dimension(3.0)  # float but whole
    3
    >>> verify_spatial_dimension(np.array([1])[0])  # numpy.int64
    1
    """
    try:
        # Must be strictly positive
        check_values(spatial_dimension, negative=False, zero=False)

        # Other type than int may be acceptable if it's a whole number
        if spatial_dimension % 1 != 0:
            raise ValueError("Spatial dimension must be a whole number!")

        # Make sure that it's an int (whole number checked must come first!)
        spatial_dimension = int(spatial_dimension)

    except TypeError as err:
        custom_message = "Invalid type for spatial dimension!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = (
            f"{spatial_dimension} is invalid for spatial dimension!"
        )
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return spatial_dimension


def verify_poly_degree(poly_degree: int) -> int:
    """Verify if the value of a given polynomial degree is valid.

    Parameters
    ----------
    poly_degree : int
        Polynomial degree to verify; the value of a polynomial degree must be
        non-negative (>= 0). ``poly_degree`` may not necessarily be
        an `int` but it must be a single whole number.

    Returns
    -------
    int
        Verified polynomial degree. If the input is not an `int`,
        the function does a type conversion to an `int` if possible.

    Raises
    ------
    TypeError
        If ``poly_degree`` is not of a correct type, i.e., its
        non-negativeness cannot be verified or the conversion to `int`
        cannot be carried out.
    ValueError
        If ``poly_degree`` is, for example, not a positive
        or a whole number.

    Examples
    --------
    >>> verify_poly_degree(0)  # int
    0
    >>> verify_poly_degree(1.0)  # float but whole
    1
    >>> verify_poly_degree(np.array([2])[0])  # numpy.int64
    2
    """
    try:
        # Must be non-negative
        check_values(poly_degree, negative=False)

        # Other type than int may be acceptable if it's a whole number
        if poly_degree % 1 != 0:
            raise ValueError("Poly. degree must be a whole number!")

        # Make sure that it's an int (whole number checked must come first!)
        poly_degree = int(poly_degree)

    except TypeError as err:
        custom_message = "Invalid type for poly. degree!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = f"{poly_degree} is invalid for poly. degree! "
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return poly_degree


def verify_lp_degree(lp_degree: float) -> float:
    """Verify that the value of a given lp-degree is valid.

    Parameters
    ----------
    lp_degree : float
        A given :math:`p` of the :math:`l_p`-norm (i.e., :math:`l_p`-degree)
        to verify. The value of an ``lp_degree`` must be strictly positive, but
        may not necessarily be a `float`.

    Returns
    -------
    float
        Verified lp-degree value. If the input is not a `float`, the function
        does a type conversion to a `float` if possible.

    Raises
    ------
    TypeError
        If ``lp_degree`` is not of correct type, i.e., its strict-positiveness
        cannot be verified or the conversion to `float` cannot be carried
        out.
    ValueError
        If ``lp-degree`` is, for example, a non strictly positive value.

    Examples
    --------
    >>> verify_lp_degree(2.5)  # float
    2.5
    >>> verify_lp_degree(3)  # int
    3.0
    >>> verify_lp_degree(np.array([1])[0])  # numpy.int64
    1.0
    """
    try:
        # Must be strictly positive, infinity is allowed
        check_values(lp_degree, inf=True, negative=False, zero=False)

        # Make sure that it's a float
        lp_degree = float(lp_degree)

    except TypeError as err:
        custom_message = "Invalid type for lp-degree!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = (
            f"{lp_degree} is invalid for lp-degree (must be > 0)!"
        )
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return lp_degree


def _add_custom_exception_message(
    exception_args: Tuple[str, ...],
    custom_message: str
) -> Tuple[str, ...]:
    """Prepend a custom message to an exception message.

    Parameters
    ----------
    exception_args : Tuple[str, ...]
        The arguments of the raised exception.
    custom_message : str
        The custom message to be prepended.

    Returns
    -------
    Tuple[str, ...]
        Modified exception arguments.
    """
    if not exception_args:
        arg = custom_message
    else:
        arg = f"{exception_args[0]} {custom_message}"
    exception_args = (arg,) + exception_args[1:]

    return exception_args


if __name__ == "__main__":
    import doctest
    doctest.testmod()
