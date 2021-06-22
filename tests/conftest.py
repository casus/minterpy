"""
This is the conftest module of minterpy.

Within a pytest run, this module is loaded first. That means here all global fixutes shall be defined.
"""

import pytest
import numpy as np

from numpy.testing import assert_equal,assert_almost_equal,assert_

from minterpy import MultiIndex
# Global seed
SEED = 12345678


# asserts that a call runs as expected
def assert_call(fct,*args,**kwargs):
    try:
        fct(*args,**kwargs)
    except Exception as e:
        print(type(e))
        raise AssertionError(f"The function was not called properly. It raised the exception:\n\n {e.__class__.__name__}: {e}")


#assert if multi_indices are equal
def assert_multi_index_equal(mi1,mi2):
    try:
        assert_equal(mi1.exponents,mi2.exponents)
        assert_equal(mi1.lp_degree,mi2.lp_degree)
        assert_equal(mi1.poly_degree,mi2.poly_degree)
    except AssertionError as a:
        raise AssertionError(f"The two instances of MultiIndex are not equal:\n\n {a}")

#assert if multi_indices are almost equal
def assert_multi_index_almost_equal(mi1,mi2):
    try:
        assert_almost_equal(mi1.exponents,mi2.exponents)
        assert_almost_equal(mi1.lp_degree,mi2.lp_degree)
        assert_almost_equal(mi1.poly_degree,mi2.poly_degree)
    except AssertionError as a:
        raise AssertionError(f"The two instances of MultiIndex are not almost equal:\n\n {a}")

#assert if polynomials are almost equal
def assert_polynomial_equal(P1,P2):
    try:
        assert_(isinstance(P1,type(P2)))
        assert_multi_index_equal(P1.multi_index,P2.multi_index)
        assert_equal(P1.coeffs,P2.coeffs)
    except AssertionError as a:
        raise AssertionError(f"The two instances of {P1.__class__.__name__} are not equal:\n\n {a}")

#assert if polynomials are almost equal
def assert_polynomial_almost_equal(P1,P2):
    try:
        assert_(isinstance(P1,type(P2)))
        assert_multi_index_almost_equal(P1.multi_index,P2.multi_index)
        assert_almost_equal(P1.coeffs,P2.coeffs)
    except AssertionError as a:
        raise AssertionError(f"The two instances of {P1.__class__.__name__} are not almost equal:\n\n {a}")




# fixtures for spatial dimension

spatial_dimensions = [1,3]

@pytest.fixture(params = spatial_dimensions)
def SpatialDimension(request):
    return request.param

# fixture for polynomial degree

polynomial_degree = [1,3]

@pytest.fixture(params = polynomial_degree)
def PolyDegree(request):
    return request.param

# fixture for lp degree

lp_degree = [1,2,np.inf]

@pytest.fixture(params = lp_degree)
def LpDegree(request):
    return request.param

# fixtures for multi_indices

@pytest.fixture()
def MultiIndices(SpatialDimension,PolyDegree,LpDegree):
    return MultiIndex.from_degree(SpatialDimension,PolyDegree,LpDegree)

# fixtures for number of similar polynomials

nr_similar_polynomials = [None,1,2]

@pytest.fixture(params = nr_similar_polynomials)
def NrSimilarPolynomials(request):
    return request.param


# some random builder

def build_rnd_exponents(dim,n,seed = None):
    """Build random exponents.

    For later use, if ``MultiIndex`` will accept arbitrary exponents again.

    :param dim: spatial dimension
    :param n: number of random monomials

    Notes
    -----
    Exponents are generated within the intervall ``[MIN_POLY_DEG,MAX_POLY_DEG]``

    """
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    return np.random.randint(MIN_POLY_DEG,MAX_POLY_DEG,(n,dim))

def build_rnd_coeffs(mi,nr_poly=None,seed = None):
    """Build random coefficients.

    For later use.

    :param mi: The :class:`MultiIndex` instance of the respective polynomial.
    :type mi: MultiIndex

    :param nr_poly: Number of similar polynomials. Default is 1
    :type nr_poly: int, optional

    :return: Random array of shape ``(nr of monomials,nr of similar polys)`` usable as coefficients.
    :rtype: np.ndarray

    """
    if nr_poly is None:
        additional_coeff_shape = tuple()
    else:
        additional_coeff_shape = (nr_poly,)
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    return np.random.random((len(mi),)+ additional_coeff_shape)


def build_rnd_points(nr_points,spatial_dimension,nr_poly=None,seed = None):
    """Build random coefficients.

    For later use.

    :param mi: The :class:`MultiIndex` instance of the respective polynomial.
    :type mi: MultiIndex

    :param nr_poly: Number of similar polynomials. Default is 1
    :type nr_poly: int, optional

    :return: Random array of shape ``(nr of monomials,nr of similar polys)`` usable as coefficients.
    :rtype: np.ndarray

    """
    if nr_poly is None:
        additional_coeff_shape = tuple()
    else:
        additional_coeff_shape = (nr_poly,)
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    return np.random.random((nr_points,spatial_dimension)+ additional_coeff_shape)
