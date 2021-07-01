"""
testing module for the polynomial submodule.

Here only the execution/initialization is tested, not the correct values. For value testing, see the testing module of the concrete implementations.
"""
import pytest
from conftest import assert_call
import numpy as np
from numpy.testing import assert_

from minterpy import CanonicalPolynomial, NewtonPolynomial, LagrangePolynomial,MultiIndexSet
from minterpy.core.ABC import MultivariatePolynomialSingleABC


classes = [CanonicalPolynomial, NewtonPolynomial, LagrangePolynomial]
class_ids =  tuple(cls.__name__ for cls in classes)

@pytest.fixture(params=classes, ids=class_ids)
def Polynom(request):
    return request.param


# test init

def test_init_polynomials(Polynom):
    assert_(issubclass(Polynom,MultivariatePolynomialSingleABC))

    mi = MultiIndexSet.from_degree(2,1,1)
    coeffs = np.arange(len(mi),dtype=float)
    assert_call(Polynom,mi,coeffs)
