"""
Testing module for interpolation.py

Here the functionality of the respective attribute is not tested.
"""

import numpy as np
import pytest
from conftest import (LpDegree, MultiIndices, NrPoints, PolyDegree,
                      SpatialDimension, assert_call, assert_grid_equal,
                      assert_interpolant_almost_equal,
                      assert_multi_index_equal, assert_polynomial_almost_equal,
                      build_random_newton_polynom, build_rnd_points)
from numpy.testing import assert_, assert_almost_equal

import minterpy as mp
from minterpy import Interpolant, Interpolator, interpolate

# test construction


def test_init_interpolator(SpatialDimension, PolyDegree, LpDegree):
    assert_call(Interpolator, SpatialDimension, PolyDegree, LpDegree)
    interpolator = Interpolator(SpatialDimension, PolyDegree, LpDegree)
    groundtruth_multi_index = mp.MultiIndexSet.from_degree(
        SpatialDimension, PolyDegree, LpDegree
    )
    groundtruth_grid = mp.Grid(groundtruth_multi_index)
    assert_multi_index_equal(interpolator.multi_index, groundtruth_multi_index)
    assert_grid_equal(interpolator.grid, groundtruth_grid)


def test_init_interpolant(SpatialDimension, PolyDegree, LpDegree):
    assert_call(
        Interpolant,
        lambda x: x[:, 0],
        Interpolator(SpatialDimension, PolyDegree, LpDegree),
    )
    assert_call(
        Interpolant.from_degree,
        lambda x: x[:, 0],
        SpatialDimension,
        PolyDegree,
        LpDegree,
    )
    interpolant_default = Interpolant(
        lambda x: x[:, 0], Interpolator(SpatialDimension, PolyDegree, LpDegree)
    )
    interpolant_from_degree = Interpolant.from_degree(
        lambda x: x[:, 0], SpatialDimension, PolyDegree, LpDegree
    )
    assert_interpolant_almost_equal(interpolant_default, interpolant_from_degree)


def test_call_interpolate(SpatialDimension, PolyDegree, LpDegree):
    assert_call(interpolate, lambda x: x[:, 0], SpatialDimension, PolyDegree, LpDegree)
    interpolant = interpolate(lambda x: x[:, 0], SpatialDimension, PolyDegree, LpDegree)
    assert_(isinstance(interpolant, Interpolant))


# test if the interpolator can interpolate
def test_interpolator(SpatialDimension, PolyDegree, LpDegree):
    groundtruth_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )
    interpolator = Interpolator(SpatialDimension, PolyDegree, LpDegree)
    res_from_newton_poly = interpolator(groundtruth_poly)
    res_from_canonical_poly = interpolator(mp.NewtonToCanonical(groundtruth_poly)())
    assert_polynomial_almost_equal(res_from_newton_poly, groundtruth_poly)
    assert_polynomial_almost_equal(res_from_canonical_poly, groundtruth_poly)


# test if the interpolant interpolates
def test_interpolant(NrPoints, SpatialDimension, PolyDegree, LpDegree):
    rnd_points = build_rnd_points(NrPoints, SpatialDimension)
    groundtruth_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )
    groundtruth = groundtruth_poly(rnd_points)
    interpolant = Interpolant.from_degree(
        groundtruth_poly, SpatialDimension, PolyDegree, LpDegree
    )
    res = interpolant(rnd_points)
    assert_almost_equal(res, groundtruth)


# test if the interpolate does what it promisses
def test_interpolate(NrPoints, SpatialDimension, PolyDegree, LpDegree):
    rnd_points = build_rnd_points(NrPoints, SpatialDimension)
    groundtruth_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )
    groundtruth = groundtruth_poly(rnd_points)
    interpolant = interpolate(groundtruth_poly, SpatialDimension, PolyDegree, LpDegree)
    res = interpolant(rnd_points)
    assert_almost_equal(res, groundtruth)
