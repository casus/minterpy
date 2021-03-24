import unittest
from itertools import product

import numpy as np

from tests.auxiliaries import all_are_close, check_different_settings, get_derivator, almost_equal, \
    get_multi_index, rnd_points
from minterpy import LagrangePolynomial, CanonicalPolynomial, NewtonPolynomial, get_transformation, compute_grad_c2c, \
    compute_grad_x2c
from minterpy.derivation import partial_derivative_canonical, derive_gradient_canonical, tensor_right_product, \
    tensor_left_product
from minterpy.global_settings import FLOAT_DTYPE
from minterpy.multi_index_utils import is_lexicographically_complete
from minterpy.verification import check_shape

NR_SPEED_SAMPLES = int(1e2)
# DO_SPEED_TESTS = True
DO_SPEED_TESTS = False

NR_NUMERICAL_SAMPLES = int(1e2)
# DO_NUMERICAL_TESTS = True
DO_NUMERICAL_TESTS = False


def test_canonical_gradient():
    print('\ntesting gradient construction...')
    # ATTENTION: the exponent vectors of all derivatives have to be included already!
    exponents = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2]])
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert exponents.shape == (5, 3)
    assert coeffs.shape == (5,)

    grad = derive_gradient_canonical(coeffs, exponents)
    grad_expected = np.array([[ 0.,  2.,  3.],
                              [ 0.,  0.,  4.],
                              [ 0.,  4., 10.],
                              [ 0.,  0.,  0.],
                              [ 0.,  0.,  0.]])
    assert grad.shape == exponents.shape, f'unexpected gradient shape: {grad.shape}'
    almost_equal(grad, grad_expected), f"unexpected gradient: {grad}"
    print('tests passed!')


def check_grad_basis_transformations(spatial_dimension, poly_degree, lp_degree):
    # compute the 'base' gradient operator (canonical basis -> "c2c")
    multi_index = get_multi_index(spatial_dimension, poly_degree, lp_degree)
    exponents = multi_index.exponents
    nr_monomials, dimensionality = exponents.shape
    assert is_lexicographically_complete(exponents)

    # Expected shape:
    grad_op_c2c = np.zeros((dimensionality, nr_monomials, nr_monomials), dtype=FLOAT_DTYPE)
    # equal to construction with passing identity matrices  as transformations
    compute_grad_c2c(grad_op_c2c, exponents)

    # compute the gradient operator for any in any basis (-> "x2x")
    poly_classes = [CanonicalPolynomial, LagrangePolynomial, NewtonPolynomial]
    for cls_from, cls_to in product(poly_classes, repeat=2):
        derivator = get_derivator(spatial_dimension, poly_degree, lp_degree, cls_from=cls_from,
                                  cls_to=cls_to)
        d = derivator._derivator
        grad_op = d._gradient_op
        origin_poly = d.origin_poly
        assert type(origin_poly) is cls_from
        # 'manually' compute the gradient operator "x2x" based on the basis operator
        x2c = get_transformation(origin_poly, CanonicalPolynomial).transformation
        grad_op_x2c = tensor_right_product(grad_op_c2c, x2c)

        canonical_poly = CanonicalPolynomial(None, d.multi_index)
        c2x = get_transformation(canonical_poly, cls_to).transformation
        grad_op_x2x = tensor_left_product(c2x, grad_op_x2c)

        almost_equal(grad_op_x2x, grad_op)


def check_equality_to_canonical_grad_op(spatial_dimension, poly_degree, lp_degree):
    # tests the basic canonical 2 canonical gradient operator computation
    multi_index = get_multi_index(spatial_dimension, poly_degree, lp_degree)
    exponents = multi_index.exponents
    nr_monomials, dimensionality = exponents.shape
    assert is_lexicographically_complete(exponents)

    # Expected shape:
    grad_op1 = np.zeros((dimensionality, nr_monomials, nr_monomials), dtype=FLOAT_DTYPE)
    grad_op2 = grad_op1.copy()

    # equal to construction with passing identity matrices  as transformations
    c2c = np.eye(nr_monomials)
    compute_grad_x2c(grad_op1, exponents, c2c)
    compute_grad_c2c(grad_op2, exponents)

    derivator = get_derivator(spatial_dimension, poly_degree, lp_degree, cls_from=CanonicalPolynomial,
                              cls_to=CanonicalPolynomial)
    grad_op3 = derivator._derivator._gradient_op
    all_are_close([grad_op1, grad_op2, grad_op3])


def check_gradient_analytical(spatial_dimension, poly_degree, lp_degree):
    # degree is fixed
    n = poly_degree
    assert n == 2
    m = spatial_dimension
    # define a (scalar) quadratic function for which we already know the derivative (=gradient)
    derivator = get_derivator(m, n, lp_degree, cls_from=LagrangePolynomial,
                              cls_to=LagrangePolynomial)
    lagr_poly = derivator._derivator.origin_poly
    assert type(lagr_poly) is LagrangePolynomial
    grid = lagr_poly.grid
    unisolvent_nodes = grid.unisolvent_nodes

    a = rnd_points(m)
    A = np.diag(a)
    b = rnd_points(1, m)
    c = rnd_points(1)
    f = lambda x: x.T @ A @ x + b @ x + c
    fx = lambda x: 2 * x.T @ A + b
    # evaluate f as well as its derivative fx at the Chebyshev nodes of our polynomial:
    f_vals = np.apply_along_axis(f, 1, unisolvent_nodes)
    fx_vals = np.apply_along_axis(fx, 1, unisolvent_nodes).squeeze()
    # .reshape((-1, m),)

    # set the coefficients of the polynomial which should be derived:
    coeffs_lagrange = f_vals.reshape(-1)
    lagr_poly.coeffs = coeffs_lagrange

    # compute the gradient
    grad_lagrange = derivator.get_gradient_poly()
    coeffs_grad_lagr = grad_lagrange.coeffs.squeeze()
    # the lagrange coefficients of the gradient polynomial (= values of the gradient at the unisolvent nodes)
    # should be equal to the values of the analytical derivative
    almost_equal(fx_vals, coeffs_grad_lagr)

    # # TODO test operator computation
    # # compute the gradient of our polynomial at each grid_point (equal to gradient in Lagrange basis)
    # l2c = derivator.lagrange2canonical
    # c2l = derivator.canonical2lagrange
    # grad_op_l2l = get_lagrange_gradient_operator(l2c, c2l,
    #                                              derivator.exponents)
    #
    # # the lagrange coefficients of the gradient (partial derivatives) are the polynomial values
    # # at the corresponding interpolation nodes
    # grad_lagrange = get_gradient(coeffs_lagrange, grad_op_l2l)


# # TODO test!
# def test_lagrange_monomial_gradient_eval(m, n):
#     print('\ntesting the evaluation of Lagrange monomial gradient...')
#     regressor = MultivariatePolynomialRegression(m, n)
#     point = rnd_points(m)
#     # should automatically initialise gradient!
#     grad_eval = regressor.eval_lagrange_monomial_gradient_on(point)
#     m_grad, N, _ = regressor.grad_op_l2n.shape
#     assert regressor.N_fit == N
#     assert m == m_grad
#     grad_eval_ref = np.empty((m, N))
#     for i in range(m):
#         gradient_coeffs = regressor.grad_op_l2n[i]
#         grad_eval_ref[i] = regressor.transformer.tree.eval(point, gradient_coeffs)
#     np.testing.assert_allclose(grad_eval_ref, grad_eval)


class DerivatorTest(unittest.TestCase):

    def test_equality_to_canonical_grad_op(self):
        print('\ntesting the basic gradient operator tensor construction...')
        check_different_settings(check_equality_to_canonical_grad_op)

    def test_tensor_product(self):
        print('\ntesting tensor product implementation...')
        check_different_settings(check_grad_basis_transformations)

    def test_gradient_analytical(self):
        print('\ntesting gradient computation with an analytical example:')
        # example is of fixed polynomial degree 2:
        check_different_settings(check_gradient_analytical, test_degrees=[2])

    def test_partial_derivation_canonical(self):
        print('\ntesting partial derivation...')
        # ATTENTION: the exponent vectors of all derivatives have to be included already! (completeness!)
        exponents = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2]])
        check_shape(exponents, (5, 3))
        assert is_lexicographically_complete(exponents)
        coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        check_shape(coeffs, (5,))

        # TODO just partial derivative, also test other functions!
        coeffs_canonical_deriv0 = partial_derivative_canonical(0, coeffs, exponents)
        # coefficients should become 0.0
        almost_equal(coeffs_canonical_deriv0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

        coeffs_canonical_deriv1 = partial_derivative_canonical(1, coeffs, exponents)
        # coefficients should "change places"
        almost_equal(coeffs_canonical_deriv1, np.array([2.0, 0.0, 4.0, 0.0, 0.0]))

        coeffs_canonical_deriv2 = partial_derivative_canonical(2, coeffs, exponents)
        # NOTE: coefficients should be multiplied with the exponent
        almost_equal(coeffs_canonical_deriv2, np.array([3.0, 4.0, 10.0, 0.0, 0.0]))
        print('tests passed!')

    # TODO test Joint Polynonial derivation

    # # TODO
    # def test_accuracy(self):
    #     print('\ntesting the gradient accuracy')
    #     check_different_settings(gradient accuracy_test)


if __name__ == '__main__':
    # suite = unittest.TestLoader().loadTestsFromTestCase(HelperTest)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
