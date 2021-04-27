# -*- coding:utf-8 -*-

import time
import unittest
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from auxiliaries import rnd_points, check_different_settings, get_grid
from minterpy import MultiIndex, LagrangePolynomial, TransformationLagrangeToNewton, Grid, \
    TransformationNewtonToLagrange
from minterpy.utils import report_error
from test_settings import DESIRED_PRECISION, NR_SAMPLE_POINTS, TIME_FORMAT_STR, RUNGE_FCT_VECTORIZED
# TODO more sophisticated tests
# TODO test tree!
# test if interpolation is globally converging
# test grid structure
from transformation_test import check_poly_interpolation


def accuracy_test_fct(spatial_dimension, poly_degree, lp_degree):
    t1 = time.time()
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    print("Indices generated in", TIME_FORMAT_STR.format(time.time() - t1))

    t1 = time.time()
    grid = Grid(multi_index)
    interpolation_nodes = grid.unisolvent_nodes
    print("Grid generated in", TIME_FORMAT_STR.format(time.time() - t1))

    t1 = time.time()
    ground_truth_fct = RUNGE_FCT_VECTORIZED  # TODO test different functions
    fct_values = ground_truth_fct(interpolation_nodes)  # evaluate ground truths function
    print("Groundtruth generated in", TIME_FORMAT_STR.format(time.time() - t1))

    nr_coefficients = len(multi_index)
    print(f"  - no. coefficients: {nr_coefficients}")
    coeffs_lagrange = fct_values
    lagrange_poly = LagrangePolynomial(coeffs_lagrange, multi_index, grid=grid)
    l2n_transformation = TransformationLagrangeToNewton(lagrange_poly)

    # estimate parameters using lagrange2newton transformation
    t1 = time.time()
    # NOTE: includes building the transformation matrix!
    newton_poly = l2n_transformation()

    # coeffs_newton = newton_poly.coeffs
    print("l2n transformation (=interpolation) took", TIME_FORMAT_STR.format(time.time() - t1))

    # test transformation:
    # the Lagrange coefficients (on the interpolation grid)
    # should be equal to the function values (on the interpolation grid)
    n2l_transformation = TransformationNewtonToLagrange(newton_poly)
    lagrange_poly2 = n2l_transformation()
    coeffs_lagrange2 = lagrange_poly2.coeffs
    np.testing.assert_allclose(coeffs_lagrange, coeffs_lagrange2)

    # If the interpolation works correctly the interpolating polynomial ("interpolant")
    # evaluated at the interpolation nodes has equal values to the interpolated function!
    # -> Evaluate the interpolant at all interpolation points (sanity check)
    # error should be at machine precision
    vals_interpol = newton_poly(interpolation_nodes)
    err = fct_values - vals_interpol
    report_error(err, 'error at interpolation nodes (accuracy check):')
    np.testing.assert_almost_equal(vals_interpol, fct_values, decimal=DESIRED_PRECISION)

    # Evaluate polynomial on uniformly sampled points
    pts_uniformly_random = rnd_points(NR_SAMPLE_POINTS, spatial_dimension)
    vals_interpol = newton_poly(pts_uniformly_random)
    vals_true = ground_truth_fct(pts_uniformly_random)
    err = vals_true - vals_interpol
    report_error(err, f'error on {NR_SAMPLE_POINTS} uniformly random points in [-1,1]^m')

    if spatial_dimension != 2:
        return
    # Evaluate polynomial on equidistant grid
    # TODO option to specify density!
    x = np.arange(-1, 1, step=0.1)
    y = np.arange(-1, 1, step=0.1)
    x, y = np.meshgrid(x, y)
    equidist_grid = np.stack([x.reshape(-1), y.reshape(-1)], axis=1)

    vals_interpol = newton_poly(equidist_grid)
    vals_true = ground_truth_fct(equidist_grid)
    err = vals_true - vals_interpol
    report_error(err, 'error on equidistant grid')

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(vals_true.reshape([len(x), len(x)]))
    ax1.set_title("ground truths")
    ax2.imshow(vals_interpol.reshape([len(x), len(x)]))
    ax2.set_title("interpolant")
    plt.show()


def check_grid_enlarge(spatial_dimension, poly_degree, lp_degree):
    grid = get_grid(spatial_dimension, poly_degree, lp_degree)
    grid_enlarged = grid.enlarge()
    # the unisolvent nodes stay unchanged
    nodes = grid.unisolvent_nodes
    nodes_enlarged = grid_enlarged.unisolvent_nodes
    # NOTE: the generating values in use might slightly mismatch due to numerical errors
    np.testing.assert_array_almost_equal(nodes, nodes_enlarged)
    # the multi index SHAPE stays unchanged
    assert grid.multi_index.exponents.shape == grid_enlarged.multi_index.exponents.shape
    # the generating points should become larger (corresponding to a higher degree)
    generating_points = grid.generating_points
    generating_points_enlarged = grid_enlarged.generating_points
    assert generating_points.shape[0] <= generating_points_enlarged.shape[0]
    generating_values = grid.generating_values
    generating_values_enlarged = grid_enlarged.generating_values
    assert len(generating_values) <= len(generating_values_enlarged)
    # TODO test more thoroughly. test if remapping of the indices worked
    #  -> exponents must point to the same grid value!


def interpolate_ground_truth_poly(spatial_dimension, poly_degree, lp_degree):
    grid = get_grid(spatial_dimension, poly_degree, lp_degree)
    check_poly_interpolation(grid)


class MainPackageTest(unittest.TestCase):

    # test settings:
    # setting1 = False

    @classmethod
    def setUpClass(cls):
        # preparations which have to be made only once
        pass

    # TEST CASES:
    # NOTE: all test case names have to start with "test..."
    def test_correctness(self):
        print('\n\ntesting interpolation of ground truth polynomials:\n')
        check_different_settings(interpolate_ground_truth_poly)

    def test_accuracy(self):
        print('\n\ntesting runge function interpolation (accuracy test):\n')
        check_different_settings(accuracy_test_fct)

    def test_grid_enlarge(self):
        check_different_settings(check_grid_enlarge)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", dest="m", type=int, help="input dimension", default=2)
    parser.add_argument("-n", dest="n", type=int, help="polynomial degree", default=15)
    parser.add_argument("-lp_degree", dest="lp_degree", type=float, help="LP order", default=2)
    args = parser.parse_args()

    n = args.n
    m = args.m
    lp_deg = args.lp

    assert m > 0, "m must be larger than 0"
    accuracy_test_fct(m, n, lp_deg)
