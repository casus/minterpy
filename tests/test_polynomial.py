"""
testing module for the polynomial submodule.

Here only the execution/initialization is tested, not the correct values. For value testing, see the testing module of the concrete implementations.
"""
import numpy as np
import pytest
from conftest import assert_call
from numpy.testing import assert_

from minterpy import (
    CanonicalPolynomial,
    LagrangePolynomial,
    MultiIndexSet,
    NewtonPolynomial,
)
from minterpy.core.ABC import MultivariatePolynomialSingleABC

classes = [CanonicalPolynomial, NewtonPolynomial, LagrangePolynomial]
class_ids = tuple(cls.__name__ for cls in classes)


@pytest.fixture(params=classes, ids=class_ids)
def Polynom(request):
    return request.param


# test init


def test_init_polynomials(Polynom):
    assert_(issubclass(Polynom, MultivariatePolynomialSingleABC))

    mi = MultiIndexSet.from_degree(2, 1, 1)
    coeffs = np.arange(len(mi), dtype=float)
    assert_call(Polynom, mi, coeffs)


@pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
def test_empty_set(spatial_dimension, Polynom, LpDegree):
    """Test construction with an empty set."""
    # Create an empty set
    mi = MultiIndexSet(np.empty((0, spatial_dimension)), LpDegree)

    # Assertion
    with pytest.raises(ValueError):
        Polynom(mi)