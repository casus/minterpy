import math
import numpy as np
import pytest

from scipy.sparse.linalg import gmres
from numpy.testing import assert_

import minterpy as mp
from minterpy.extras.regression import OrdinaryRegression
from minterpy.polynomials import (
    LagrangePolynomial, NewtonPolynomial, CanonicalPolynomial
)

from conftest import (
    SpatialDimension,
    LpDegree,
    PolyDegree,
    assert_call,
    assert_multi_index_equal,
    assert_grid_equal,
    build_random_newton_polynom,
)


def _solve_gmres(rr: np.ndarray, yy: np.ndarray, ww: np.ndarray, **kwargs):
    """Solve a least-squares problem using a Generalized Min. Resid. solver."""
    if ww is None:
        coeffs, _ = gmres(rr, yy, atol="legacy", **kwargs)
    else:
        coeffs, _ = gmres(
            rr.T @ ww @ rr, rr.T @ ww @ yy, atol="legacy", **kwargs
        )

    return coeffs


# Fixtures for least-squares solver keyword arguments (These are the defaults)
LSQ_SOLVER_ARGS = {
    "lstsq": {"cond": None, "check_finite": True},
    "pinv": {"rcond": 1e-15},
    "dgesv": {"overwrite_a": False},
    "dsysv": {"lower": False},
    "dposv": {"overwrite_b": False},
}


def assert_uninitialized_polynomial_equal(poly_1, poly_2):
    try:
        assert_(isinstance(poly_1, type(poly_2)))
        assert_multi_index_equal(poly_1.multi_index, poly_2.multi_index)
        with pytest.raises(ValueError):
            _ = poly_1.coeffs
            _ = poly_2.coeffs
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {poly_1.__class__.__name__} "
            f"are not equal:\n\n {a}"
        )


# Fixtures for least-squares solver
least_squares_solvers = [
    "lstsq",
    "inv",
    "pinv",
    "dgesv",
    "dsysv",
    "dposv",
    "qr",
    "svd",
    _solve_gmres,
]


@pytest.fixture(params=least_squares_solvers)
def least_squares_solver(request):
    return request.param


# Fixtures for least-squares solver
origin_polys = [
    LagrangePolynomial, NewtonPolynomial, CanonicalPolynomial
]


@pytest.fixture(params=origin_polys)
def origin_poly(request):
    return request.param


def test_ordinary_regression_init_multi_index(
        SpatialDimension, PolyDegree, LpDegree
):
    """Test the initialization."""
    # Create a multi-index set
    multi_index_ref = mp.MultiIndexSet.from_degree(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create a grid from the multi-index set
    grid_ref = mp.Grid(multi_index_ref)

    # Create an instance of origin poly
    origin_poly_ref = mp.LagrangePolynomial(multi_index_ref)

    # Create an instance using obligatory parameters
    my_ordinary_regression = OrdinaryRegression(multi_index_ref)

    # Multi-index attribute is correctly assigned
    assert_multi_index_equal(
        my_ordinary_regression.multi_index, multi_index_ref
    )

    # Grid attribute is correctly assigned
    assert_grid_equal(my_ordinary_regression.grid, grid_ref)

    # Origin polynomial is correctly instantiated
    assert_uninitialized_polynomial_equal(
        my_ordinary_regression.origin_poly, origin_poly_ref
    )

    # Assert uninitialized attributes prior to fitting
    assert my_ordinary_regression.loocv_error is None
    assert my_ordinary_regression.regfit_linf_error is None
    assert my_ordinary_regression.regfit_l2_error is None
    assert my_ordinary_regression.coeffs is None
    assert my_ordinary_regression.eval_poly is None

    # show() method can be called
    assert_call(my_ordinary_regression.show)


def test_ordinary_regression_init_grid(
        SpatialDimension, PolyDegree, LpDegree, origin_poly
):
    """Test the initialization."""
    # Create a multi-index set
    multi_index_ref = mp.MultiIndexSet.from_degree(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create a grid from the multi-index set
    grid_ref = mp.Grid(multi_index_ref)

    # Initializer can be called only with a grid instance
    my_ordinary_regression = OrdinaryRegression(grid=grid_ref)

    # Multi-index attribute is correctly assigned
    assert_multi_index_equal(
        my_ordinary_regression.multi_index, multi_index_ref
    )

    # Grid attribute is correctly assigned
    assert_grid_equal(my_ordinary_regression.grid, grid_ref)

    # Assert uninitialized attributes prior to fitting
    assert my_ordinary_regression.loocv_error is None
    assert my_ordinary_regression.regfit_linf_error is None
    assert my_ordinary_regression.regfit_l2_error is None
    assert my_ordinary_regression.coeffs is None
    assert my_ordinary_regression.eval_poly is None


def test_ordinary_regression_init_grid_multi_index(
    SpatialDimension,
    PolyDegree,
    LpDegree,
    origin_poly,
):
    """Test the initialization."""
    # Create a multi-index set
    multi_index_ref = mp.MultiIndexSet.from_degree(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create a grid from the multi-index set
    grid_ref = mp.Grid(multi_index_ref)

    # Create an instance of origin polynomial
    origin_poly_ref = origin_poly(multi_index=multi_index_ref, grid=grid_ref)

    # Initializer can be called with different origin poly basis
    my_ordinary_regression = OrdinaryRegression(
        multi_index_ref, grid=grid_ref, origin_poly=origin_poly
    )

    # Multi-index attribute is correctly assigned
    assert_multi_index_equal(
        my_ordinary_regression.multi_index, multi_index_ref
    )

    # Grid attribute is correctly assigned
    assert_grid_equal(my_ordinary_regression.grid, grid_ref)

    # Origin polynomial is correctly instantiated
    assert_uninitialized_polynomial_equal(
        my_ordinary_regression.origin_poly, origin_poly_ref
    )

    # Assert uninitialized attributes prior to fitting
    assert my_ordinary_regression.loocv_error is None
    assert my_ordinary_regression.regfit_linf_error is None
    assert my_ordinary_regression.regfit_l2_error is None
    assert my_ordinary_regression.coeffs is None
    assert my_ordinary_regression.eval_poly is None


def test_ordinary_regression_fit(
    SpatialDimension,
    PolyDegree,
    LpDegree,
    least_squares_solver,
    origin_poly,
):
    """Test if a fit can be carried out on the instance."""
    # Create a multi-index set
    multi_index = mp.MultiIndexSet.from_degree(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create a test training set from a random Newton polynomial
    newton_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )
    xx_train = newton_poly.grid.unisolvent_nodes
    yy_train = newton_poly(xx_train)

    # Create an OrdinaryRegression instance
    my_ordinary_regression = OrdinaryRegression(
        multi_index, origin_poly=origin_poly
    )

    # fit() can be called
    # NOTE: Regression on the unisolvent nodes with training points at the
    #       unisolvent nodes is interpolation
    my_ordinary_regression.fit(xx_train, yy_train)
    assert np.allclose(yy_train, my_ordinary_regression(xx_train))

    # Test fit() call with a vector of weights
    weights = np.ones(yy_train.shape)
    assert_call(my_ordinary_regression.fit, xx_train, yy_train, weights=weights)

    # Test fit() call with a matrix of weights
    weights = np.eye(yy_train.shape[0])
    assert_call(my_ordinary_regression.fit, xx_train, yy_train, weights=weights)

    # Different least-squares solver
    assert_call(
        my_ordinary_regression.fit,
        xx_train,
        yy_train,
        weights=weights,
        lstsq_solver=least_squares_solver
    )

    # fit() can be called with additional fitting arguments
    kwargs = LSQ_SOLVER_ARGS.get(least_squares_solver, {})
    assert_call(
        my_ordinary_regression.fit,
        xx_train,
        yy_train,
        lstsq_solver=least_squares_solver,
        **kwargs
    )

    # Assert uninitialized attributes prior to fitting
    assert my_ordinary_regression.loocv_error is not None
    assert my_ordinary_regression.regfit_linf_error is not None
    assert my_ordinary_regression.regfit_l2_error is not None
    assert my_ordinary_regression.coeffs is not None
    assert my_ordinary_regression.eval_poly is not None

    # show() method can be called
    assert_call(my_ordinary_regression.show)


def test_ordinary_regression_fit_not_implemented_solver(
    SpatialDimension,
    PolyDegree,
    LpDegree,
):
    """Test if a fit with not implemented solver raises an error."""
    # Create a multi-index set
    multi_index = mp.MultiIndexSet.from_degree(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create a test training set from a random Newton polynomial
    newton_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )
    xx_train = newton_poly.grid.unisolvent_nodes
    yy_train = newton_poly(xx_train)

    # Create an OrdinaryRegression instance
    my_ordinary_regression = OrdinaryRegression(multi_index)

    with pytest.raises(NotImplementedError):
        my_ordinary_regression.fit(xx_train, yy_train, lstsq_solver="rand123")


def test_ordinary_regression_predict(
    SpatialDimension,
    PolyDegree,
    LpDegree,
    origin_poly,
):
    """Test if the prediction can be carried out on the instance."""
    # Create a multi-index set
    multi_index = mp.MultiIndexSet.from_degree(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create the corresponding grid
    grid = mp.Grid(multi_index)

    # Create a test training set from a random Newton polynomial
    newton_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )
    xx_train = newton_poly.grid.unisolvent_nodes
    yy_train = newton_poly(xx_train)

    # Create and fit an OrdinaryRegression instance
    my_ordinary_regression = OrdinaryRegression(
        multi_index=multi_index, grid=grid, origin_poly=origin_poly
    )

    # predict() without fitting
    with pytest.raises(TypeError):
        my_ordinary_regression.predict(xx_train)  # predict method
        my_ordinary_regression(xx_train)          # __call__ dunder method

    # Fit the ordinary regression model
    my_ordinary_regression.fit(xx_train, yy_train)

    # predict() can be called
    assert_call(my_ordinary_regression.predict, xx_train)  # predict method
    assert_call(my_ordinary_regression, xx_train)          # __call__ method

    # Test the prediction on a test dataset
    xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
    yy_test = newton_poly(xx_test)
    assert np.allclose(yy_test, my_ordinary_regression(xx_test))


def test_wrong_multi_index_instance():
    """Test if passing a wrong instance of multi-index raises an error."""

    mi = mp.MultiIndexSet.from_degree(1, 3, 1)
    mi_exponents = mi.exponents

    with pytest.raises(TypeError):
        OrdinaryRegression(mi_exponents)


def test_wrong_grid_instance():
    """Test if passing a wrong instance of grid raises an error."""

    mi = mp.MultiIndexSet.from_degree(2, 3, 1)

    with pytest.raises(TypeError):
        OrdinaryRegression(grid=mi)


def test_insufficient_input():
    """Test if passing an insufficient input to constructor raises an error."""

    with pytest.raises(ValueError):
        OrdinaryRegression()


def test_grid_subset_of_multi_index():
    """Test if passing a grid of lower degree than multi-index raises an error.
    """

    # Grid is of higher degree than the polynomial multi-index: okay
    mi_poly = mp.MultiIndexSet.from_degree(2, 3, 1)
    mi_grid = mp.MultiIndexSet.from_degree(2, 5, 1)
    grid = mp.Grid(mi_grid)

    assert_call(OrdinaryRegression, multi_index=mi_poly, grid=grid)

    # Grid is of lower degree than the polynomial multi-index: not okay
    mi_grid = mp.MultiIndexSet.from_degree(2, 2, 1)
    grid = mp.Grid(mi_grid)

    with pytest.raises(ValueError):
        OrdinaryRegression(mi_poly, grid)


def test_underdetermined_system(SpatialDimension, PolyDegree, LpDegree):
    """Test fitting under-determined system."""

    # Create a multi-index set
    multi_index = mp.MultiIndexSet.from_degree(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create a test training set from a random Newton polynomial
    newton_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # At least one training point
    sample_size = max(1, math.floor(0.8 * len(multi_index)))
    xx_train = -1 + 2 * np.random.rand(sample_size, SpatialDimension)
    yy_train = newton_poly(xx_train)

    # Create and fit an OrdinaryRegression instance
    my_ordinary_regression = OrdinaryRegression(multi_index)

    # Fit the ordinary regression model
    my_ordinary_regression.fit(xx_train, yy_train, lstsq_solver="qr")

    assert my_ordinary_regression.loocv_error == (np.infty, np.infty)


def test_single_monomial(origin_poly):
    """Test an edge case of a single monomial."""

    spatial_dimension = 1
    poly_degree = 0
    lp_degree = 1
    # Create a multi-index set
    multi_index = mp.MultiIndexSet.from_degree(
        spatial_dimension, poly_degree, lp_degree
    )

    # Create a test training set from a random Newton polynomial
    newton_poly = build_random_newton_polynom(
        spatial_dimension, poly_degree, lp_degree
    )
    xx_train = -1 + 2 * np.random.rand(100, 1)
    yy_train = newton_poly(xx_train)

    # Create and fit an OrdinaryRegression instance
    my_ordinary_regression = OrdinaryRegression(
        multi_index=multi_index, origin_poly=origin_poly
    )

    # predict() without fitting
    with pytest.raises(TypeError):
        my_ordinary_regression.predict(xx_train)  # predict method
        my_ordinary_regression(xx_train)  # __call__ dunder method

    # Fit the ordinary regression model
    my_ordinary_regression.fit(xx_train, yy_train)

    # predict() can be called
    assert_call(my_ordinary_regression.predict, xx_train)  # predict method
    assert_call(my_ordinary_regression, xx_train)  # __call__ method

    # Test the prediction on a test dataset
    xx_test = -1 + 2 * np.random.rand(1000, spatial_dimension)
    yy_test = newton_poly(xx_test)
    assert np.allclose(yy_test, my_ordinary_regression(xx_test))


def test_unsupported_polynomial_basis():
    """Test fitting an unsupported polynomial basis."""
    spatial_dimension = 1
    poly_degree = 3
    lp_degree = 2.0
    # Create a multi-index set
    multi_index = mp.MultiIndexSet.from_degree(
        spatial_dimension, poly_degree, lp_degree
    )

    # Create a test training set
    xx_train = -1 + 2 * np.random.rand(100, 1)
    yy_train = 2 * xx_train

    # Create and fit an OrdinaryRegression instance
    my_ordinary_regression = OrdinaryRegression(
        multi_index=multi_index, origin_poly=_EmptyClass
    )

    with pytest.raises(TypeError):
        my_ordinary_regression.fit(xx_train, yy_train)


def test_compute_errors():
    """Test the flag for LOO CV error computations."""
    spatial_dimension = 3
    poly_degree = 3
    lp_degree = 2.0
    # Create a multi-index set
    multi_index = mp.MultiIndexSet.from_degree(
        spatial_dimension, poly_degree, lp_degree
    )

    # Create a test training set
    xx_train = -1 + 2 * np.random.rand(100, 3)
    yy_train = 2 * xx_train[:,0]

    # Create and fit an OrdinaryRegression instance
    my_ordinary_regression = OrdinaryRegression(multi_index=multi_index)

    # Fit and compute LOO-CV error
    my_ordinary_regression.fit(xx_train, yy_train, compute_loocv=True)
    assert my_ordinary_regression.loocv_error is not None
    assert_call(my_ordinary_regression.show)

    # Fit but don't compute LOO-CV error
    my_ordinary_regression.fit(xx_train, yy_train, compute_loocv=False)
    assert my_ordinary_regression.loocv_error is None
    assert_call(my_ordinary_regression.show)

    # Fit with invalid value for the compute_loocv flag
    with pytest.raises(ValueError):
        my_ordinary_regression.fit(xx_train, yy_train, compute_loocv=100)


@pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
def test_empty_set(spatial_dimension, LpDegree):
    """Test construction with an empty set."""
    # Create an empty set
    mi = mp.MultiIndexSet(np.empty((0, spatial_dimension)), LpDegree)

    # Assertion
    with pytest.raises(ValueError):
        OrdinaryRegression(mi)


class _EmptyClass:
    def __init__(self, multi_index, grid):
        self.multi_index = multi_index
        self.grid = grid
        self.exponents = None
        self.generating_points = None
