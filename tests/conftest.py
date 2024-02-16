"""
This is the conftest module of minterpy.

Within a pytest run, this module is loaded first. That means here all global fixutes shall be defined.
"""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_, assert_almost_equal, assert_equal

from minterpy import MultiIndexSet, NewtonPolynomial

# Global seed
SEED = 12345678

# Global settings
MIN_POLY_DEG = 0
MAX_POLY_DEG = 25

# asserts that a call runs as expected
def assert_call(fct, *args, **kwargs):
    try:
        fct(*args, **kwargs)
    except Exception as e:
        print(type(e))
        raise AssertionError(
            f"The function was not called properly. It raised the exception:\n\n {e.__class__.__name__}: {e}"
        )


# assert if multi_indices are equal
def assert_multi_index_equal(mi1, mi2):
    try:
        assert_(isinstance(mi1, type(mi2)))
        assert_equal(mi1.exponents, mi2.exponents)
        assert_equal(mi1.lp_degree, mi2.lp_degree)
        assert_equal(mi1.poly_degree, mi2.poly_degree)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of MultiIndexSet are not equal:\n\n {a}"
        )


# assert if multi_indices are almost equal
def assert_multi_index_almost_equal(mi1, mi2):
    try:
        assert_(isinstance(mi1, type(mi2)))
        assert_almost_equal(mi1.exponents, mi2.exponents)
        assert_almost_equal(mi1.lp_degree, mi2.lp_degree)
        assert_almost_equal(mi1.poly_degree, mi2.poly_degree)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of MultiIndexSet are not almost equal:\n\n {a}"
        )


# assert if two grids are equal
def assert_grid_equal(grid1, grid2):
    try:
        assert_(isinstance(grid1, type(grid2)))
        assert_equal(grid1.unisolvent_nodes, grid2.unisolvent_nodes)
        assert_equal(grid1.spatial_dimension, grid2.spatial_dimension)
        assert_equal(grid1.generating_values, grid2.generating_values)
        assert_equal(grid1.generating_values, grid2.generating_values)
        assert_multi_index_equal(grid1.multi_index, grid2.multi_index)
    except AssertionError as a:
        raise AssertionError(f"The two instances of Grid are not equal:\n\n {a}")


# assert if two grids are almost equal
def assert_grid_almost_equal(grid1, grid2):
    try:
        assert_(isinstance(grid1, type(grid2)))
        assert_almost_equal(grid1.unisolvent_nodes, grid2.unisolvent_nodes)
        assert_almost_equal(grid1.spatial_dimension, grid2.spatial_dimension)
        assert_almost_equal(grid1.generating_values, grid2.generating_values)
        assert_almost_equal(grid1.generating_values, grid2.generating_values)
        assert_multi_index_almost_equal(grid1.multi_index, grid2.multi_index)
    except AssertionError as a:
        raise AssertionError(f"The two instances of Grid are not almost equal:\n\n {a}")


# assert if polynomials are almost equal
def assert_polynomial_equal(P1, P2):
    try:
        assert_(isinstance(P1, type(P2)))
        assert_multi_index_equal(P1.multi_index, P2.multi_index)
        assert_equal(P1.coeffs, P2.coeffs)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {P1.__class__.__name__} are not equal:\n\n {a}"
        )


# assert if polynomials are almost equal
def assert_polynomial_almost_equal(P1, P2):
    try:
        assert_(isinstance(P1, type(P2)))
        assert_multi_index_almost_equal(P1.multi_index, P2.multi_index)
        assert_almost_equal(P1.coeffs, P2.coeffs)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {P1.__class__.__name__} are not almost equal:\n\n {a}"
        )


# assert if two functions have the same object code
def assert_function_object_code_equal(fct1, fct2):
    try:
        assert_(fct1.__code__.co_code == fct2.__code__.co_code)
    except AssertionError as a:
        raise AssertionError(
            f"The object_code of {fct1} is not equal to the object_code of {fct2}:\n\n {inspect.getsource(fct1)}\n\n{inspect.getsource(fct2)}"
        )


# assert if interpolators are equal
def assert_interpolator_equal(interpolator1, interpolator2):
    try:
        assert_(isinstance(interpolator1, type(interpolator2)))
        assert_equal(interpolator1.spatial_dimension, interpolator2.spatial_dimension)
        assert_equal(interpolator1.poly_degree, interpolator2.poly_degree)
        assert_equal(interpolator1.lp_degree, interpolator2.lp_degree)
        assert_multi_index_equal(interpolator1.multi_index, interpolator2.multi_index)
        assert_grid_equal(interpolator1.grid, interpolator2.grid)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {interpolator1.__class__.__name__} are not equal:\n\n {a}"
        )


# assert if interpolators are almost equal
def assert_interpolator_almost_equal(interpolator1, interpolator2):
    try:
        assert_(isinstance(interpolator1, type(interpolator2)))
        assert_almost_equal(
            interpolator1.spatial_dimension, interpolator2.spatial_dimension
        )
        assert_almost_equal(interpolator1.poly_degree, interpolator2.poly_degree)
        assert_almost_equal(interpolator1.lp_degree, interpolator2.lp_degree)
        assert_multi_index_almost_equal(
            interpolator1.multi_index, interpolator2.multi_index
        )
        assert_grid_almost_equal(interpolator1.grid, interpolator2.grid)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {interpolator1.__class__.__name__} are not almost equal:\n\n {a}"
        )


# assert if interpolants are equal
def assert_interpolant_equal(interpolant1, interpolant2):
    try:
        assert_(isinstance(interpolant1, type(interpolant2)))
        assert_function_object_code_equal(interpolant1.fct, interpolant2.fct)
        assert_interpolator_equal(interpolant1.interpolator, interpolant2.interpolator)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {interpolant1.__class__.__name__} are not equal:\n\n {a}"
        )


# assert if interpolants are almost equal
def assert_interpolant_almost_equal(interpolant1, interpolant2):
    try:
        assert_(isinstance(interpolant1, type(interpolant2)))
        assert_function_object_code_equal(interpolant1.fct, interpolant2.fct)
        assert_interpolator_almost_equal(
            interpolant1.interpolator, interpolant2.interpolator
        )
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {interpolant1.__class__.__name__} are not almost equal:\n\n {a}"
        )


# fixtures for spatial dimension

spatial_dimensions = [1, 3]


@pytest.fixture(params=spatial_dimensions)
def SpatialDimension(request):
    return request.param


# fixture for polynomial degree
# NOTE: Include test for poly_degree 0 (Issue #27)
polynomial_degree = [0, 1, 4]


@pytest.fixture(params=polynomial_degree)
def PolyDegree(request):
    return request.param


# fixture for lp degree

lp_degree = [0.5, 1, 2, np.inf]


@pytest.fixture(params=lp_degree)
def LpDegree(request):
    return request.param


# fixtures for multi_indices


@pytest.fixture()
def MultiIndices(SpatialDimension, PolyDegree, LpDegree):
    return MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)


# fixtures for number of similar polynomials

nr_similar_polynomials = [None, 1, 2]


@pytest.fixture(params=nr_similar_polynomials)
def NrSimilarPolynomials(request):
    return request.param


# fixture for the number of points for evaluations

nr_pts = [1, 2]


@pytest.fixture(params=nr_pts)
def NrPoints(request):
    return request.param


# Fixture for the number
nr_polynomials = [1, 10]


@pytest.fixture(params=nr_polynomials)
def NrPolynomials(request):
    return request.param


# Fixture for the number
batch_sizes = [1, 100, 1000]


@pytest.fixture(params=batch_sizes)
def BatchSizes(request):
    return request.param


# Fixture for pair
@pytest.fixture(
    params=[
        "equal",
        "dimensions",
        "poly-degrees",
        "lp-degrees",
        "empty",
        "all",
    ]
)
def param_diff(request):
    return request.param


@pytest.fixture
def mi_pair(SpatialDimension, PolyDegree, LpDegree, param_diff):
    """Create a pair of MultiIndexSets with different parameters."""
    if param_diff == "equal":
        # A pair with equal parameter values
        m = SpatialDimension
        n = PolyDegree
        p = LpDegree
        mi_1 = MultiIndexSet.from_degree(m, n, p)
        mi_2 = MultiIndexSet.from_degree(m, n, p)
    elif param_diff == "dimensions":
        # A pair with different spatial dimensions.
        m_1 = SpatialDimension
        m_2 = SpatialDimension + 1
        n = PolyDegree
        p = LpDegree
        mi_1 = MultiIndexSet.from_degree(m_1, n, p)
        mi_2 = MultiIndexSet.from_degree(m_2, n, p)
    elif param_diff == "poly-degrees":
        # A pair with different polynomial degrees.
        m = SpatialDimension
        n_1 = PolyDegree
        n_2 = PolyDegree + np.random.randint(low=1, high=3)
        p = LpDegree
        mi_1 = MultiIndexSet.from_degree(m, n_1, p)
        mi_2 = MultiIndexSet.from_degree(m, n_2, p)
    elif param_diff == "lp-degrees":
        # A pair with different lp-degrees.
        m = SpatialDimension
        d = PolyDegree
        lp_degrees = [0.5, 1.0, 2.0, 3.0, np.inf]
        p_1, p_2 = np.random.choice(lp_degrees, size=2, replace=False)
        mi_1 = MultiIndexSet.from_degree(m, d, p_1)
        mi_2 = MultiIndexSet.from_degree(m, d, p_2)
    elif param_diff == "all":
        # A pair with all three parameters differ
        m_1 = np.random.randint(low=1, high=5)
        m_2 = np.random.randint(low=1, high=5)
        d_1 = np.random.randint(low=1, high=5)
        d_2 = np.random.randint(low=1, high=5)
        lp_degrees = [0.5, 1.0, 2.0, 3.0, np.inf]
        p_1, p_2 = np.random.choice(lp_degrees, 2)
        mi_1 = MultiIndexSet.from_degree(m_1, d_1, p_1)
        mi_2 = MultiIndexSet.from_degree(m_2, d_2, p_2)
    elif param_diff == "empty":
        # A pair with one of them is empty
        m_1 = np.random.randint(low=1, high=5)
        m_2 = np.random.randint(low=1, high=5)
        d_1 = np.random.randint(low=1, high=5)
        lp_degrees = [0.5, 1.0, 2.0, 3.0, np.inf]
        p_1, p_2 = np.random.choice(lp_degrees, 2)
        mi_1 = MultiIndexSet.from_degree(m_1, d_1, p_1)
        mi_2 = MultiIndexSet(np.empty((0, m_2)), p_2)
    else:
        return ValueError(f"'param-diff' = {param_diff} is not recognized!")

    return mi_1, mi_2

# some random builder


def build_rnd_exponents(dim, n, seed=None):
    """Build random exponents.

    For later use, if ``MultiIndexSet`` will accept arbitrary exponents again.

    :param dim: spatial dimension
    :param n: number of random monomials

    Notes
    -----
    Exponents are generated within the intervall ``[MIN_POLY_DEG,MAX_POLY_DEG]``

    """
    rng = np.random.default_rng(seed)

    exponents = rng.integers(MIN_POLY_DEG, MAX_POLY_DEG, (n, dim), dtype=int)

    return exponents


def build_rnd_coeffs(mi, nr_poly=None, seed=None):
    """Build random coefficients.

    For later use.

    :param mi: The :class:`MultiIndexSet` instance of the respective polynomial.
    :type mi: MultiIndexSet

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
    return np.random.random((len(mi),) + additional_coeff_shape)


def build_rnd_points(nr_points, spatial_dimension, nr_poly=None, seed=None):
    """Build random points in space.

    Return a batch of `nr_points` vectors in the real vector space of dimension ``spatial_dimension``.

    :param nr_points: Number of points
    :type nr_points: int

    :param spatial_dimension: Dimension of domain space.
    :type spatial_dimension: int

    :param nr_poly: Number of similar polynomials. Default is 1
    :type nr_poly: int, optional

    :param seed: Seed used for the random generation. Default is SEED.
    :type seed: int

    :return: Array of shape ``(nr_points,spatial_dimension[,nr_poly])`` containig random real values (distributed in the intervall :math:`[-1,1]`).
    :rtype: np.ndarray

    """
    if nr_poly is None:
        additional_coeff_shape = tuple()
    else:
        additional_coeff_shape = (nr_poly,)
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    return np.random.uniform(
        -1, 1, size=(nr_points, spatial_dimension) + additional_coeff_shape
    )


def build_random_newton_polynom(
    dim: int, deg: int, lp: int,  n_poly=1, seed=None
) -> NewtonPolynomial:
    """Build a random Newton polynomial.

    Return a :class:`NewtonPolynomial` with a lexicographically complete :class:`MultiIndex` (initiated from ``(dim,deg,lp)``) and randomly generated coefficients (uniformly distributes in the intervall :math:`[-1,1]`).

    :param dim: dimension of the domain space.
    :type dim: int
    :param deg: degree of the interpolation polynomials
    :type deg: int
    :param lp: degree of the :math:`l_p` norm used to determine the `poly_degree`.
    :type lp: int

    :return: Newton polynomial with random coefficients.
    :rtype: NewtonPolynomial

    """
    mi = MultiIndexSet.from_degree(dim, deg, lp)
    if seed is None:
        seed = SEED

    np.random.seed(seed)

    if n_poly == 1:
        rnd_coeffs = np.random.uniform(-1, 1, size=len(mi))
    else:
        rnd_coeffs = np.random.uniform(-1, 1, size=(len(mi), n_poly))

    return NewtonPolynomial(mi, rnd_coeffs)


def build_random_multi_index():
    """Build random complete multi-index set."""
    m = np.random.randint(1, 5)
    n = np.random.randint(1, 5)
    p = np.random.choice([1.0, 2.0, np.inf])

    mi = MultiIndexSet.from_degree(m, n, p)

    return mi
