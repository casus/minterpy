"""Test for the input verification module.
"""
import numpy as np
import pytest
from numpy.testing import assert_raises
from conftest import assert_call

from minterpy.core.verification import (
    verify_spatial_dimension,
    verify_lp_degree,
)


# --- verify_lp_degree()
@pytest.mark.parametrize(
    "lp_degree",
    [1, 1.0, np.array([1])[0], np.array([1.0])[0]]
)
def test_verify_lp_degree(lp_degree):
    """Test the verification of an lp-degree value."""
    assert_call(verify_lp_degree, lp_degree)

    verified_lp_degree = verify_lp_degree(lp_degree)

    # Assertions
    assert isinstance(verified_lp_degree, float)
    assert verified_lp_degree == float(lp_degree)


def test_verify_lp_degree_type_error():
    """Test raising TypeError in the lp-degree verification."""
    lp_degree = "2.0"
    assert_raises(TypeError, verify_lp_degree, lp_degree)


@pytest.mark.parametrize(
    "lp_degree",
    [0, 0.0, -100, -100.0, np.array([-10])[0], np.array([1, 2, 3])]
)
def test_verify_spatial_dimension_value_error(spatial_dimension):
    """Test raising ValueError in the lp-degree verification."""
    assert_raises(ValueError, verify_spatial_dimension, spatial_dimension)


# --- verify_spatial_dimension()
@pytest.mark.parametrize(
    "spatial_dimension",
    [1, 1.0, np.array([1])[0], np.array([1.0])[0]]
)
def test_verify_spatial_dimension(spatial_dimension):
    """Test the verification of a spatial dimension value.

    Notes
    -----
    - This test is related to Issue #77.
    """
    assert_call(verify_spatial_dimension, spatial_dimension)

    verified_spatial_dimension = verify_spatial_dimension(spatial_dimension)

    # Assertions
    assert isinstance(verified_spatial_dimension, int)
    assert verified_spatial_dimension == int(spatial_dimension)


def test_verify_spatial_dimension_type_error():
    """Test raising TypeError in the spatial dimension verification.

    Notes
    -----
    - This test is related to Issue #77.
    """
    spatial_dimension = "1"
    assert_raises(TypeError, verify_spatial_dimension, spatial_dimension)


@pytest.mark.parametrize(
    "spatial_dimension",
    [0, 0.0, 1.1, -100, -100.0, np.array([-10])[0], np.array([1, 2, 3])]
)
def test_verify_spatial_dimension_value_error(spatial_dimension):
    """Test raising ValueError in the spatial dimension verification.

    Notes
    -----
    - This test is related to Issue #77.
    """
    assert_raises(ValueError, verify_spatial_dimension, spatial_dimension)
