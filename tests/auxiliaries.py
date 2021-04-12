from itertools import product
from random import randint
from typing import Iterable, List, Type

import numpy as np

from minterpy import MultiIndex, Grid, LagrangePolynomial, Derivator, MultivariatePolynomialSingleABC, \
    TransformationABC
from minterpy.transformation_meta import get_transformation_class
from minterpy.verification import check_shape, check_is_square
from tests.test_settings import DIMENSIONS2TEST, DEGREES2TEST, LP_DEGREES, DESIRED_PRECISION


def proto_test_case(data, fct):
    for input, expected_output in data:
        # print(input, expected_output, fct(input))
        actual_output = fct(input)
        if actual_output != expected_output:
            print('input: {} expected: {} got: {}'.format(input, expected_output, actual_output))
        assert actual_output == expected_output


def rnd_points(*shape):
    return 2 * (np.random.rand(*shape) - 0.5)  # [-1;1]


def print_test_setttings(spatial_dimension, poly_degree, lp_degree):
    print("----------------------------"
          f"\n- dimensionality: {spatial_dimension}"
          f"\n- degree: {poly_degree}"
          f"\n- lp_degree: {lp_degree}")


def check_different_settings(test_fct, test_degrees: Iterable[int] = DEGREES2TEST, ):
    # for g_vectorized in TEST_FUNCTIONS_VECTORIZED:
    for spatial_dimension, poly_degree, lp_degree in product(DIMENSIONS2TEST, test_degrees, LP_DEGREES):
        print_test_setttings(spatial_dimension, poly_degree, lp_degree)
        test_fct(spatial_dimension, poly_degree, lp_degree)
        print('PASSED.')


def almost_equal(a, b):
    np.testing.assert_almost_equal(a, b, decimal=DESIRED_PRECISION)


def check_is_identity(a):
    check_is_square(a)
    size = a.shape[0]
    almost_equal(a, np.eye(size))


def check_transformation_is_inverse(a2b, b2a):
    b2a2b = np.dot(b2a, a2b)
    check_is_identity(b2a2b)

    a2b2a = np.dot(a2b, b2a)
    check_is_identity(a2b2a)


def all_are_close(arrays: List[np.ndarray]):
    a0 = arrays.pop()
    for a in arrays:
        almost_equal(a0, a)


def all_have_shape(array_iter: Iterable[np.ndarray], shape, dimensionality: int = None):
    for a in array_iter:
        check_shape(a, shape, dimensionality)


def delete_rnd_exp(exponents):
    nr_indices = exponents.shape[0]
    rnd_idx = randint(0, nr_indices - 1)
    exponents = np.delete(exponents, rnd_idx, axis=0)
    return exponents


def delete_random_index(multi_index):
    nr_indices = len(multi_index)
    if nr_indices > 1:  # at least one entry needs to remain
        exponents = multi_index.exponents
        exponents = delete_rnd_exp(exponents)
        multi_index = MultiIndex(exponents)
    return multi_index


def make_incomplete(multi_index):
    # delete indices until the index set is incomplete
    for i in range(len(multi_index) - 1):
        if not multi_index.is_complete:
            break
        multi_index = delete_random_index(multi_index)
    return multi_index


def get_multi_index(spatial_dimension, poly_degree, lp_degree, get_incomplete: bool = False) -> MultiIndex:
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    if get_incomplete:
        multi_index = make_incomplete(multi_index)
    return multi_index


def get_grid(spatial_dimension, poly_degree, lp_degree, get_incomplete: bool = False) -> Grid:
    multi_index = get_multi_index(spatial_dimension, poly_degree, lp_degree, get_incomplete)
    return Grid(multi_index)


def get_poly(spatial_dimension, poly_degree, lp_degree,
             cls: Type[MultivariatePolynomialSingleABC] = LagrangePolynomial,
             get_incomplete: bool = False, separate_indices: bool = False) -> MultivariatePolynomialSingleABC:
    grid = get_grid(spatial_dimension, poly_degree, lp_degree, get_incomplete)
    multi_index = grid.multi_index
    if separate_indices:
        multi_index = delete_random_index(multi_index)
    return cls(None, multi_index, grid=grid)


def get_transformation(spatial_dimension, poly_degree, lp_degree, cls_from=LagrangePolynomial,
                       cls_to=LagrangePolynomial, get_incomplete: bool = False,
                       separate_indices: bool = False) -> TransformationABC:
    poly = get_poly(spatial_dimension, poly_degree, lp_degree, cls=cls_from, get_incomplete=get_incomplete,
                    separate_indices=separate_indices)
    transformer_cls = get_transformation_class(cls_from, cls_to)
    return transformer_cls(origin_poly=poly)


def get_derivator(spatial_dimension, poly_degree, lp_degree, cls_from=LagrangePolynomial,
                  cls_to=LagrangePolynomial, get_incomplete: bool = False,
                  separate_indices: bool = False) -> Derivator:
    poly = get_poly(spatial_dimension, poly_degree, lp_degree, cls=cls_from, get_incomplete=get_incomplete,
                    separate_indices=separate_indices)
    return Derivator(origin_poly=poly, target_type=cls_to)
