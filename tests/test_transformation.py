"""

testing module for Transformation classes.

"""

import pytest
import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_raises
from conftest import (assert_call, assert_polynomial_almost_equal, build_rnd_coeffs,
                        SpatialDimension, PolyDegree, LpDegree)

from minterpy import (MultiIndexSet, CanonicalPolynomial, NewtonPolynomial, LagrangePolynomial)
from minterpy.core.ABC import TransformationABC,OperatorABC
from minterpy.transformations import (LagrangeToNewton, NewtonToLagrange,
                                      LagrangeToCanonical, CanonicalToLagrange,
                                      NewtonToCanonical, CanonicalToNewton, Identity,
                                      get_transformation, get_transformation_class)

from minterpy.transformations.utils import (_build_lagrange_to_newton_naive, _build_lagrange_to_newton_bary,
                                            _build_newton_to_lagrange_naive, _build_newton_to_lagrange_bary,
                                            build_l2n_matrix_dds)

transform_classes = [LagrangeToNewton, NewtonToLagrange,
                     LagrangeToCanonical, CanonicalToLagrange,
                     NewtonToCanonical, CanonicalToNewton]

@pytest.fixture(params=transform_classes)
def Transform(request):
    return request.param


def test_init_transform(Transform):
    """testing the initialization of transformation classes"""

    assert_(issubclass(Transform,TransformationABC))

    # Test initialization
    mi = MultiIndexSet.from_degree(2, 2, 1.0)
    coeffs = np.arange(len(mi), dtype=float)
    poly = Transform.origin_type(mi, coeffs)
    assert_call(Transform, poly)

    # test transformation call
    transform = Transform(poly)
    assert_call(transform)

    # test the type of transformed poly
    res_poly = transform()
    assert_(isinstance(res_poly, Transform.target_type))

    # test existence of transformation_operator
    operator = transform.transformation_operator
    assert_(isinstance(operator, OperatorABC))

poly_classes = [CanonicalPolynomial, NewtonPolynomial, LagrangePolynomial]

@pytest.fixture(params=poly_classes)
def Polynom(request):
    return request.param

P1 = Polynom
P2 = Polynom

def test_get_transformation(P1, P2):
    """ test the get_transformation function in transformation_meta"""

    mi = MultiIndexSet.from_degree(2, 1, 1.0)
    coeffs = np.arange(len(mi), dtype=float)
    poly = P1(mi, coeffs)

    transform = get_transformation(poly, P2)

    if P1 == P2:
        assert_(isinstance(transform, Identity))
    else:
        assert_(isinstance(transform, TransformationABC))

def test_fail_get_transformation_class():
    """ tests if get_transformation_class throws an error if it cannot find a transforamtion"""

    assert_raises(NotImplementedError, get_transformation_class, None, LagrangePolynomial)

def test_l2n_transform(SpatialDimension, PolyDegree, LpDegree):
    """ testing the naive and bary centric l2n transformations """
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    coeffs = build_rnd_coeffs(mi)
    lag_poly = LagrangePolynomial(mi, coeffs)

    transformation_l2n = LagrangeToNewton(lag_poly)

    # test naive
    transform_naive = _build_lagrange_to_newton_naive(transformation_l2n)
    newt_coeffs_naive = transform_naive @ lag_poly.coeffs
    newt_poly_naive = NewtonPolynomial(mi, newt_coeffs_naive)

    # test bary dict
    transform_bary = _build_lagrange_to_newton_bary(transformation_l2n)
    newt_coeffs_bary = transform_bary @ lag_poly.coeffs
    newt_poly_bary = NewtonPolynomial(mi, newt_coeffs_bary)

    # compare the result of naive and bary transformation
    assert_polynomial_almost_equal(newt_poly_naive, newt_poly_bary)

    # compare the naive transformation matrix with dds constructed
    l2n_matrix = build_l2n_matrix_dds(lag_poly.grid)
    assert_almost_equal(l2n_matrix, transform_naive.array_repr_full)

    # check if newton polynomials evaluated on the unisolvent nodes are indeed lagrange coeffs
    res_eval = newt_poly_naive(lag_poly.unisolvent_nodes)
    assert_almost_equal(res_eval, lag_poly.coeffs)


def test_n2l_transform(SpatialDimension, PolyDegree, LpDegree):
    """ testing the naive and bary centric l2n transformations """
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    coeffs = build_rnd_coeffs(mi)
    newt_poly = NewtonPolynomial(mi, coeffs)

    transformation_n2l = NewtonToLagrange(newt_poly)

    # test naive
    transform_naive = _build_newton_to_lagrange_naive(transformation_n2l)
    lag_coeffs_naive = transform_naive @ newt_poly.coeffs
    lag_poly_naive = LagrangePolynomial(mi, lag_coeffs_naive)

    # test bary dict
    transform_bary = _build_newton_to_lagrange_bary(transformation_n2l)
    lag_coeffs_bary = transform_bary @ newt_poly.coeffs
    lag_poly_bary = LagrangePolynomial(mi, lag_coeffs_bary)

    # compare the result of naive and bary transformation
    assert_polynomial_almost_equal(lag_poly_naive, lag_poly_bary)


def test_transformation_identity():

    mi = MultiIndexSet.from_degree(2, 1, 1.0)
    coeffs = build_rnd_coeffs(mi)
    newt_poly = NewtonPolynomial(mi, coeffs)

    assert_call(Identity, newt_poly)
    transform = Identity(newt_poly)

    transform_mat = transform.transformation_operator.array_repr_full

    # check whether transformation matrix is identity
    assert_almost_equal(transform_mat, np.eye(len(mi)))

    assert_call(transform)
    res_poly = transform()

    # check the resulting polynomial is same after transformation
    assert_polynomial_almost_equal(res_poly, newt_poly)


def test_transform_back_n_forth(P1,P2,SpatialDimension, PolyDegree, LpDegree):
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    coeffs = build_rnd_coeffs(mi)
    origin_poly = P1(mi, coeffs)

    transform_forward = get_transformation(origin_poly,P2)
    interim_poly = transform_forward()

    transform_back = get_transformation(interim_poly,P1)
    final_poly = transform_back()

    assert_polynomial_almost_equal(origin_poly, final_poly)
