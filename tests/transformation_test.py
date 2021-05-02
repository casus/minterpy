# -*- coding:utf-8 -*-
import unittest

import numpy as np
import scipy.linalg

from auxiliaries import check_different_settings, rnd_points, almost_equal, get_transformation, check_is_identity, \
    check_transformation_is_inverse, get_grid, get_separate_indices_poly
from minterpy import MultiIndex, Grid, NewtonPolynomial, \
    TransformationNewtonToCanonical, TransformationCanonicalToNewton, TransformationLagrangeToNewton, \
    TransformationNewtonToLagrange, LagrangePolynomial, TransformationABC
from minterpy.barycentric_precomp import _build_lagrange_to_newton_bary, _build_newton_to_lagrange_bary
from minterpy.global_settings import FLOAT_DTYPE, INT_DTYPE, ARRAY
from minterpy.transformation_operator_abstract import TransformationOperatorABC
from minterpy.transformation_utils import build_l2n_matrix_dds, _build_newton_to_lagrange_naive
from minterpy.utils import report_error


def check_trafo_general(transformation):
    """ performs the checks valid for all transformations
    """
    # all transformations must have a transformation operator
    operator = transformation.transformation_operator
    assert isinstance(operator, TransformationOperatorABC)

    # all transformation operators must have an array representation
    transformation_matrix = operator.array_repr_sparse
    assert isinstance(transformation_matrix, ARRAY)

    # there are as many "active" Lagrange polynomials as there are index vectors
    # the matrix should have as many columns as there are active monomials
    nr_active_monomials = len(transformation.multi_index)
    assert transformation_matrix.shape[1] == nr_active_monomials


def check_l2n_matrix(l2n_matrix, grid):
    multi_index = grid.multi_index
    # the L2N transformation matrix can be interpreted as the newton coefficients
    # of all Lagrange polynomials of this basis (given by the grid)
    newt_coeffs_lagr_mons = l2n_matrix
    # NOTE: define a "multitude" of polynomials (multiple sets of coefficients)
    newt_polys = NewtonPolynomial(newt_coeffs_lagr_mons, multi_index, grid=grid)
    # evaluating all the Lagrange polynomials on the interpolation nodes should yield the identity matrix
    # TODO this should also hold for incomplete indices!
    interpolation_nodes = grid.unisolvent_nodes
    transformation_matrix = newt_polys(interpolation_nodes)
    check_is_identity(transformation_matrix)


def check_n_l_matrices(n2l_transformation: TransformationABC):
    check_trafo_general(n2l_transformation)

    grid = n2l_transformation.origin_poly.grid
    multi_index = grid.multi_index

    newton_to_lagrange_eval = n2l_transformation.transformation_operator.array_repr_sparse
    lagrange_to_newton = scipy.linalg.inv(newton_to_lagrange_eval)
    check_l2n_matrix(lagrange_to_newton, grid)

    if multi_index.is_complete:  # dds requires complete multi indices
        lagrange_to_newton_dds = build_l2n_matrix_dds(grid)
        check_l2n_matrix(lagrange_to_newton_dds, grid)

        newton_to_lagrange_dds = scipy.linalg.inv(lagrange_to_newton_dds)
        almost_equal(lagrange_to_newton, lagrange_to_newton_dds)
        almost_equal(newton_to_lagrange_eval, newton_to_lagrange_dds)

        # FIXME: not working!
        # newton_to_lagrange_dds = invert_triangular(lagrange_to_newton_dds)
        # lagrange_to_newton = invert_triangular(newton_to_lagrange_eval)
        # almost_equal(newton_to_lagrange_eval, newton_to_lagrange_dds)
        # almost_equal(newton_to_lagrange_eval, newton_to_lagrange_dds)


def check_newt_lagr_poly_equality(lagrange_poly, newton_poly):
    # "interpolate" a the given Lagrange polynomial (transform it into Newton basis)
    # then transform it back and check if the results match -> transformations work
    # NOTE: these tests are also working for multi index sets with holes
    coeffs_newton_true = newton_poly.coeffs
    coeffs_lagrange = lagrange_poly.coeffs
    # estimate parameters using lagrange2newton transformation ("interpolation")
    # NOTE: includes building the transformation matrix!
    l2n_transformation = TransformationLagrangeToNewton(lagrange_poly)
    newton_poly2 = l2n_transformation()
    # test n2l transformation:
    # evaluating a newton polynomial on all interpolation points is equal to transforming this newton polynomial into
    # the Lagrange basis:
    n2l_transformation = TransformationNewtonToLagrange(newton_poly)
    lagrange_poly2 = n2l_transformation()
    lagrange_to_newton = l2n_transformation.transformation_operator.array_repr_sparse
    newton_to_lagrange = n2l_transformation.transformation_operator.array_repr_sparse
    check_transformation_is_inverse(lagrange_to_newton, newton_to_lagrange)
    coeffs_newton_estim = newton_poly2.coeffs
    err = coeffs_newton_estim - coeffs_newton_true
    report_error(err, f'error of the interpolated Newton coefficients (transformation):')
    almost_equal(coeffs_newton_estim, coeffs_newton_true)
    coeffs_lagrange2 = lagrange_poly2.coeffs
    err = coeffs_lagrange - coeffs_lagrange2
    report_error(err, f'error of the Lagrange coefficients (function values):')
    almost_equal(coeffs_lagrange, coeffs_lagrange2)


def check_poly_interpolation(grid: Grid):
    multi_index = grid.multi_index
    interpolation_nodes = grid.unisolvent_nodes
    nr_coefficients = len(multi_index)
    print(f"  - no. coefficients: {nr_coefficients}")
    coeffs_newton_true = rnd_points(nr_coefficients)  # \in [-1; 1]
    newton_poly = NewtonPolynomial(coeffs_newton_true, multi_index, grid=grid)

    # IDEA: evaluating a Newton polynomial on all the unisolvent nodes gives the Lagrange coefficients
    # this implicitly gives a transformation from Newton to Lagrange basis
    # -> create a random ground truth polynomial, perform this transformation
    # then check if the results of the interpolation (using the transformation matrices) are equal
    # NOTE: these tests are also working for multi index sets with holes
    # NOTE: at the same time this is a test for the Newton evaluation
    coeffs_lagrange = newton_poly(interpolation_nodes)
    lagrange_poly = LagrangePolynomial(coeffs_lagrange, multi_index, grid=grid)

    check_newt_lagr_poly_equality(lagrange_poly, newton_poly)


def lagrange_n_newton_matrix_test_complete(spatial_dimension, poly_degree, lp_degree):
    transformation = get_transformation(spatial_dimension, poly_degree, lp_degree, cls_from=NewtonPolynomial,
                                        cls_to=LagrangePolynomial)
    check_trafo_general(transformation)
    check_n_l_matrices(transformation)
    grid = transformation.origin_poly.grid
    check_poly_interpolation(grid)


def lagrange_n_newton_matrix_test_incomplete(spatial_dimension, poly_degree, lp_degree):
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    exponents = multi_index.exponents
    nr_exponents, m = exponents.shape

    for idx in range(nr_exponents):
        exponents_incomplete = np.delete(exponents, idx, axis=0)
        multi_index_incomplete = MultiIndex(exponents_incomplete)
        if multi_index_incomplete.is_complete:  # sometimes deleting indices does not create "holes"
            continue
        incomplete_grid = Grid(multi_index_incomplete)
        check_poly_interpolation(incomplete_grid)


def check_n2l_barycentric(spatial_dimension, poly_degree, lp_degree):
    # NOTE: these tests are NOT working for incomplete multi index sets
    transformation = get_transformation(spatial_dimension, poly_degree, lp_degree, cls_from=NewtonPolynomial,
                                        cls_to=LagrangePolynomial)
    check_trafo_general(transformation)

    grid = transformation.grid
    multi_index = grid.multi_index
    nr_coefficients = len(multi_index)

    barycentric_operator = _build_newton_to_lagrange_bary(transformation)
    n2l_barycentric = barycentric_operator.array_repr_sparse

    matrix_operator = _build_newton_to_lagrange_naive(transformation)
    n2l_regular = matrix_operator.array_repr_sparse

    almost_equal(n2l_regular, n2l_barycentric)

    coeffs_newton_true = rnd_points(nr_coefficients)  # \in [-1; 1]
    newton_poly = NewtonPolynomial(coeffs_newton_true, multi_index, grid=grid)

    # perform barycentric transformation
    coeffs_lagrange = barycentric_operator @ coeffs_newton_true
    lagrange_poly = LagrangePolynomial(coeffs_lagrange, multi_index, grid=grid)

    check_newt_lagr_poly_equality(lagrange_poly, newton_poly)


def check_l2n_barycentric(spatial_dimension, poly_degree, lp_degree):
    # NOTE: these tests are NOT working for incomplete multi index sets
    transformation = get_transformation(spatial_dimension, poly_degree, lp_degree, cls_from=LagrangePolynomial,
                                        cls_to=NewtonPolynomial)
    check_trafo_general(transformation)


    grid = transformation.grid
    multi_index = grid.multi_index
    nr_coefficients = len(multi_index)

    # check for array equality:
    l2n_regular = build_l2n_matrix_dds(grid)
    transformation_operator = _build_lagrange_to_newton_bary(transformation)
    l2n_barycentric = transformation_operator.array_repr_sparse
    almost_equal(l2n_regular, l2n_barycentric)

    coeffs_lagr_true = rnd_points(nr_coefficients)  # \in [-1; 1]
    lagrange_poly = LagrangePolynomial(coeffs_lagr_true, multi_index, grid=grid)

    # perform barycentric transformation
    coeffs_newton = transformation_operator @ coeffs_lagr_true
    newton_poly = NewtonPolynomial(coeffs_newton, multi_index, grid=grid)

    check_newt_lagr_poly_equality(lagrange_poly, newton_poly)


def canonical_newton_transformation_test(spatial_dimension, poly_degree, lp_degree):
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    grid = Grid(multi_index)

    nr_coefficients = len(multi_index)
    print(f"  - no. coefficients: {nr_coefficients}")
    coeffs_newton_true = rnd_points(nr_coefficients)  # \in [-1; 1]
    newton_poly = NewtonPolynomial(coeffs_newton_true, multi_index, grid=grid)

    n2c_transformation = TransformationNewtonToCanonical(newton_poly)
    # NOTE: includes building the transformation matrix!
    canonical_poly = n2c_transformation()

    # transform back:
    c2n_transformation = TransformationCanonicalToNewton(canonical_poly)
    # NOTE: includes building the transformation matrix!
    newton_poly2 = c2n_transformation()
    coeffs_newton_estim = newton_poly2.coeffs
    almost_equal(coeffs_newton_estim, coeffs_newton_true)
    err = coeffs_newton_estim - coeffs_newton_true
    report_error(err, f'error of the Newton coefficients (transformation):')

    n2c = n2c_transformation.transformation_operator.array_repr_sparse
    c2n = c2n_transformation.transformation_operator.array_repr_sparse
    check_transformation_is_inverse(c2n, n2c)


def check_separate_idx_transformation(spatial_dimension, poly_degree, lp_degree):
    lagr_poly = get_separate_indices_poly(spatial_dimension, poly_degree, lp_degree,cls=LagrangePolynomial)

    # just a single active Lagrange polynomial -> one coefficient
    coeffs = np.ones(1, dtype=FLOAT_DTYPE)
    lagr_poly.coeffs = coeffs

    # transform to Newton basis
    l2n_transformation = TransformationLagrangeToNewton(lagr_poly)
    newt_poly = l2n_transformation()  # this includes building the transformation matrix!

    # due to the properties of Lagrange polynomials and the constraints posed by the "basis",
    # this polynomial should be 0 on all grid points (basis) except the "active" point
    pt_position = lagr_poly.active_monomials

    grid_nodes = lagr_poly.grid.unisolvent_nodes
    base_nodes = np.delete(grid_nodes, pt_position, axis=0)
    vals_on_grid_pts = newt_poly(base_nodes)
    np.testing.assert_allclose(vals_on_grid_pts, 0.0)

    # only on the node corresponding to the "active" Lagrange polynomial, the overall polynomial should be 1
    active_pt = grid_nodes[pt_position, :]
    val_on_active_pt = newt_poly(active_pt)
    np.testing.assert_allclose(val_on_active_pt, 1.0)

    # the evaluation of the Lagrange monomials (Lagrange basis) should be equal:
    vals_on_grid_pts = lagr_poly.eval_lagrange_monomials_on(base_nodes)
    np.testing.assert_allclose(vals_on_grid_pts, 0.0)
    val_on_active_pt = lagr_poly.eval_lagrange_monomials_on(active_pt)
    np.testing.assert_allclose(val_on_active_pt, 1.0)


class TransformationTest(unittest.TestCase):

    def test_lagrange_newton(self):
        print('\ntesting Newton to Lagrange transformation matrix computation:')
        check_different_settings(lagrange_n_newton_matrix_test_complete)
        # NOTE: the actual test transformation functions are being tests in unit test
        #  (interpolation is equal to the transformation l2n)

    def test_lagrange_newton_incomplete(self):
        print('\ntesting Newton to Lagrange transformation with incomplete multi index sets:')
        check_different_settings(lagrange_n_newton_matrix_test_incomplete)

    # TODO test all different transformation formats!
    def test_newton2lagrange_barycentric(self):
        print('\ntesting the barycentric Lagrange to Newton transformation:')
        check_different_settings(check_n2l_barycentric)

    def test_lagrange2newton_barycentric(self):
        print('\ntesting the barycentric Lagrange to Newton transformation:')
        check_different_settings(check_l2n_barycentric)

    def test_canonical_newton(self):
        print('\ntesting canonical to and from Newton transformation:')
        check_different_settings(canonical_newton_transformation_test)

    def test_separate_idx_transformation(self):
        print('\ntesting Lagrange to Newton transformation for polynomials with a "separate" basis:')
        check_different_settings(check_separate_idx_transformation)
        # TODO add multiple separate points


if __name__ == '__main__':
    # suite = unittest.TestLoader().loadTestsFromTestCase(HelperTest)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
