"""
This is the conftest module of minterpy.

Within a pytest run, this module is loaded first. That means here all global fixutes shall be defined.
"""

import pytest
import numpy as np


# asserts that a call runs as expected
def assert_call(fct,*args,**kwargs):
    try:
        fct(*args,**kwargs)
    except Exception as e:
        print(type(e))
        raise AssertionError(f"The function was not called properly. It raised the exception:\n\n {e.__class__.__name__}: {e}")





# fixtures for spatial dimension

spatial_dimensions = [1,2,3]

@pytest.fixture(params = spatial_dimensions)
def SpatialDimension(request):
    return request.param

# fixture for polynomial degree

polynomial_degree = [1,2,3]

@pytest.fixture(params = polynomial_degree)
def PolyDegree(request):
    return request.param

# fixture for lp degree

lp_degree = [1,2,np.inf]

@pytest.fixture(params = lp_degree)
def LpDegree(request):
    return request.param
