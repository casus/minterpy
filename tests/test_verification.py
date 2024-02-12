"""Test for the input verification module.
"""
import numpy as np
import pytest

from minterpy.core.verification import (
    verify_spatial_dimension,
    verify_poly_degree,
    verify_lp_degree,
    check_dimensionality,
    check_shape,
    check_values,
)


class TestCheckValues:
    """All tests related to value (scalar or array) checking.

    Notes
    -----
    - These tests are related to Issue #131.
    """
    @pytest.mark.parametrize(
        "invalid_value, parameter",
        [(np.nan, "nan"), (np.inf, "inf"), (0, "zero"), (-1, "negative")],
    )
    def test_array_single_flag(self, invalid_value, parameter):
        """Test an array whether it contains invalid values."""
        # Create an array
        xx = np.empty((3, 2))
        xx[0, 0] = invalid_value

        # No exception raised
        param = {parameter: True}
        check_values(xx, **param)

        # Assertions
        param = {parameter: False}
        with pytest.raises(ValueError):
            check_values(xx, **param)

    @pytest.mark.parametrize(
        "invalid_value, parameter",
        [(np.nan, "nan"), (np.inf, "inf"), (0, "zero"), (-1, "negative")],
    )
    def test_scalar_single_flag(self, invalid_value, parameter):
        """Test a scalar whether it is invalid."""
        # No exception raised
        param = {parameter: True}
        check_values(invalid_value, **param)

        # Assertions
        param = {parameter: False}
        with pytest.raises(ValueError):
            check_values(invalid_value, **param)

    def test_array_multiple_flags(self):
        """Test an array whether it contains all invalid values."""
        # Create an array
        xx = np.empty((4, 4))
        xx[0, 0] = np.nan
        xx[1, 1] = np.inf
        xx[2, 2] = 0.0
        xx[3, 3] = -1

        # No exception raised
        params = {"nan": True, "inf": True, "negative": True, "zero": True}
        check_values(xx, **params)

        # Assertions
        params = {"nan": False, "inf": False, "negative": False, "zero": False}
        with pytest.raises(ValueError):
            check_values(xx, **params)


class TestVerifyLpDegree:
    """All tests related to the verification of lp-degree parameter."""

    @pytest.mark.parametrize(
        "lp_degree",
        [1, 1.0, np.array([1])[0], np.array([1.0])[0]],
    )
    def test_valid_lp_degree(self, lp_degree):
        """Test for valid lp-degree values (i.e., can be verified)."""
        # Verify lp-degree
        verified_lp_degree = verify_lp_degree(lp_degree)

        # Assertions
        assert isinstance(verified_lp_degree, float)
        assert verified_lp_degree == float(lp_degree)

    @pytest.mark.parametrize(
        "lp_degree",
        ["2.0", {2.0}, (2.0, ), [2.0]],  # string, set, tuple, list
    )
    def test_invalid_type_lp_degree(self, lp_degree):
        """Test raising TypeError in the lp-degree verification."""
        with pytest.raises(TypeError):
            verify_lp_degree(lp_degree)

    @pytest.mark.parametrize(
        "lp_degree",
        [0, 0.0, -100, -100.0, np.array([-10])[0], np.array([1, 2, 3])],
    )
    def test_invalid_value_lp_degree(self, lp_degree):
        """Test raising ValueError in the lp-degree verification."""
        with pytest.raises(ValueError):
            verify_spatial_dimension(lp_degree)


class TestVerifySpatialDimension:
    """All tests related to the verification of spatial dimension parameter.

    Notes
    -----
    - This test is related to Issue #77.
    """
    @pytest.mark.parametrize(
        "spatial_dimension",
        [1, 1.0, np.array([1])[0], np.array([1.0])[0]],
    )
    def test_valid_spatial_dimension(self, spatial_dimension):
        """Test for valid spatial dimension values (i.e., can be verified)."""
        # Verify spatial dimension
        verified_spatial_dimension = verify_spatial_dimension(
            spatial_dimension
        )
    
        # Assertions
        assert isinstance(verified_spatial_dimension, int)
        assert verified_spatial_dimension == int(spatial_dimension)

    @pytest.mark.parametrize(
        "spatial_dimension",
        ["1", {1}, (1, ), [1]],  # string, set, tuple, list
    )
    def test_invalid_type_spatial_dimension(self, spatial_dimension):
        """Test raising TypeError in the spatial dimension verification."""
        with pytest.raises(TypeError):
            verify_spatial_dimension(spatial_dimension)

    @pytest.mark.parametrize(
        "spatial_dimension",
        [0, 0.0, 1.1, -100, -100.0, np.array([-10])[0], np.array([1, 2, 3])],
    )
    def test_invalid_value_spatial_dimension(self, spatial_dimension):
        """Test raising ValueError in the spatial dimension verification."""
        with pytest.raises(ValueError):
            verify_spatial_dimension(spatial_dimension)


class TestVerifyPolyDegree:
    """All tests related to the verification of polynomial degree parameter.

    Notes
    -----
    - This test is related to Issue #101.
    """

    @pytest.mark.parametrize(
        "poly_degree",
        [0, 1, 1.0, np.array([1])[0], np.array([1.0])[0]],
    )
    def test_valid_poly_degree(self, poly_degree):
        """Test for valid polynomial degree values (i.e., can be verified)."""
        # Verify poly. degree
        verified_poly_degree = verify_poly_degree(poly_degree)

        # Assertions
        assert isinstance(verified_poly_degree, int)
        assert verified_poly_degree == int(poly_degree)

    @pytest.mark.parametrize(
        "poly_degree",
        ["1", {1}, (1,), [1]],  # string, set, tuple, list
    )
    def test_invalid_type_poly_degree(self, poly_degree):
        """Test raising TypeError in the polynomial degree verification."""
        with pytest.raises(TypeError):
            verify_poly_degree(poly_degree)

    @pytest.mark.parametrize(
        "poly_degree",
        [1.1, -100, -100.0, np.array([-10])[0], np.array([1, 2, 3])],
    )
    def test_verify_poly_degree_value_error(self, poly_degree):
        """Test raising ValueError in the polynomial degree verification."""
        with pytest.raises(ValueError):
            verify_poly_degree(poly_degree)


def test_check_dimensionality():
    """Test raising ValueError due to wrong dimensionality.

    Notes
    -----
    - This test is related to Issue #130.
    """
    # Repeat the test due to random nature of it
    for _ in range(5):
        # Problem setup
        dim_1, dim_2 = np.random.choice(np.arange(1, 8), size=2, replace=False)
        shape = tuple(np.random.randint(low=1, high=5, size=dim_1))
        xx = np.zeros(shape)

        # No error should be raised
        check_dimensionality(xx, dim_1)

        # Assertion
        with pytest.raises(ValueError):
            check_dimensionality(xx, dimensionality=dim_2)


def test_check_shape():
    """Test raising ValueError due to wrong shape.

    Notes
    -----
    - This test is related to Issue #130.
    """
    # Repeat the test due to random nature of it
    for _ in range(5):
        # Problem setup
        ndim = np.random.randint(low=1, high=8)
        shape_1 = tuple(np.random.randint(low=1, high=5, size=ndim))
        shape_2 = tuple(np.array(shape_1) + 1)
        shape_3 = tuple(np.random.randint(low=1, high=5, size=ndim+1))
        xx = np.zeros(shape_1)

        # No error should be raised
        check_shape(xx, shape_1)

        # Assertion
        with pytest.raises(ValueError):
            # Wrong shape
            check_shape(xx, shape=shape_2)
            # Wrong dimension
            check_shape(xx, shape=shape_3)
