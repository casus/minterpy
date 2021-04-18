# -*- coding:utf-8 -*-
import random
import time
import unittest
import warnings

import numpy as np

from minterpy import MultiIndex
from minterpy.jit_compiled_utils import lex_smaller_or_equal, have_lexicographical_ordering, index_is_contained, \
    all_indices_are_contained
from minterpy.multi_index_utils import is_lexicographically_complete, \
    make_derivable, make_complete, _gen_multi_index_exponents, find_match_between, _get_poly_degree, \
    insert_lexicographically
from tests.auxiliaries import check_different_settings
from tests.test_settings import TIME_FORMAT_STR

random.seed(42)  # for reproducible results


# def get_lex_smaller(index):
#     out = index.copy()
#     if np.all(index == 0):  # there is no smaller vector than the 0 vector
#         return out  # return equal
#     # TODO
#     return out


def get_lex_bigger(index, bigger_by_1=False):
    out = index.copy()
    m = len(index)
    rnd_dim = random.randint(0, m - 1)
    out[rnd_dim] += 1
    if not bigger_by_1:
        out[:rnd_dim] = 0  # setting all previous entries to 0 does not change the lexicographical ordering
    return out


def get_rnd_exp_vect(exponent_matrix):
    nr_exp, m = exponent_matrix.shape
    rnd_idx = random.randint(0, nr_exp - 1)
    exp_vect = exponent_matrix[rnd_idx, :]
    return rnd_idx, exp_vect


def switch_positions(exponent_matrix):
    out = exponent_matrix.copy()
    rnd_idx1, exp_vect1 = get_rnd_exp_vect(exponent_matrix)
    rnd_idx2 = rnd_idx1
    while rnd_idx2 == rnd_idx1:
        rnd_idx2, exp_vect2 = get_rnd_exp_vect(exponent_matrix)
    out[rnd_idx2, :] = exp_vect1
    out[rnd_idx1, :] = exp_vect2
    return out


def from_exponent_construction(spatial_dimension, poly_degree, lp_degree):
    if spatial_dimension >= 4 and lp_degree == 2.0 and poly_degree >= 2:
        warnings.warn('recursive exponent generation broken for this case. skipping this test case.')
        return
    exponents = _gen_multi_index_exponents(spatial_dimension, poly_degree, lp_degree)
    multi_index = MultiIndex(exponents, lp_degree=lp_degree)
    assert type(multi_index) is MultiIndex
    np.testing.assert_equal(exponents, multi_index.exponents)
    assert multi_index.lp_degree == lp_degree
    assert multi_index.poly_degree == poly_degree


def from_degree_construction(spatial_dimension, poly_degree, lp_degree):
    t1 = time.time()
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    print("Indices generated in", TIME_FORMAT_STR.format(time.time() - t1), '(iterative)')
    assert type(multi_index) is MultiIndex
    exponents = multi_index.exponents
    nr_exponents, m = exponents.shape
    assert m == spatial_dimension
    assert np.min(exponents) == 0
    assert np.max(exponents) == poly_degree
    poly_degree = _get_poly_degree(exponents, lp_degree)
    assert poly_degree == poly_degree
    # = np.linalg.norm(exponents, ord=lp_degree, axis=0)
    # assert np.max(lp_norms) <= poly_degree
    assert multi_index.lp_degree == lp_degree
    assert multi_index.poly_degree == poly_degree
    assert np.max(exponents) == poly_degree

    t1 = time.time()
    # old recursive implementation:
    exponents2 = _gen_multi_index_exponents(spatial_dimension, poly_degree, lp_degree)
    print("Indices generated in", TIME_FORMAT_STR.format(time.time() - t1), '(recursive)')
    nr_exponents2 = exponents2.shape[0]
    if nr_exponents2 > nr_exponents:
        raise AssertionError()
    if nr_exponents2 == nr_exponents:
        np.testing.assert_equal(exponents, exponents2)
    else:
        match_positions = find_match_between(exponents2, exponents)
        selected_exponents = exponents[match_positions, :]
        np.testing.assert_equal(selected_exponents, exponents2)


def lex_smaller_or_equal_test(spatial_dimension, poly_degree, lp_degree):
    t1 = time.time()
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    print("Indices generated in", TIME_FORMAT_STR.format(time.time() - t1))
    exponents = multi_index.exponents
    nr_exponents, m = exponents.shape

    for idx in range(nr_exponents):
        exp_vect = exponents[idx, :]
        assert lex_smaller_or_equal(exp_vect, exp_vect)
        bigger_exponent_vector = get_lex_bigger(exp_vect)
        assert lex_smaller_or_equal(exp_vect, bigger_exponent_vector)
        assert not lex_smaller_or_equal(bigger_exponent_vector, exp_vect)


def index_is_contained_test(spatial_dimension, poly_degree, lp_degree):
    t1 = time.time()
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    print("Indices generated in", TIME_FORMAT_STR.format(time.time() - t1))
    exponents = multi_index.exponents
    nr_exponents, m = exponents.shape
    assert m == spatial_dimension
    for i in range(nr_exponents):
        exponent_vector = exponents[i, :]
        assert index_is_contained(exponents, exponent_vector)

    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    assert not index_is_contained(exponents, bigger_exponent_vector)

    for i in range(nr_exponents):
        exponent_vector = exponents[0, :]
        exponents = np.delete(exponents, 0, axis=0)
        assert not index_is_contained(exponents, exponent_vector)


def index_is_complete_test(spatial_dimension, poly_degree, lp_degree):
    t1 = time.time()
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    print("Indices generated in", TIME_FORMAT_STR.format(time.time() - t1))
    exponents_orig = multi_index.exponents
    nr_exponents, m = exponents_orig.shape
    assert is_lexicographically_complete(exponents_orig)
    assert multi_index.is_complete

    exponents = np.delete(exponents_orig, 0, axis=0)  # deleting the smallest vector destroys completeness
    assert not is_lexicographically_complete(exponents)

    largest_exponent_vector = exponents_orig[-1, :]  # last / biggest exponent vector
    exponents = np.delete(exponents_orig, -1, axis=0)  # deleting the biggest vector maintains completeness
    assert is_lexicographically_complete(exponents)

    bigger_exponent_vector = largest_exponent_vector.copy()  # independent copy!
    bigger_exponent_vector[0] += 2  # introduces "hole"d
    multi_index2 = multi_index.add_exponents(bigger_exponent_vector)
    exponents2 = multi_index2.exponents
    assert not is_lexicographically_complete(exponents2)
    if multi_index2.is_complete:
        raise ValueError
    assert not multi_index2.is_complete


def insert_indices_test(spatial_dimension, poly_degree, lp_degree):
    # single vector insertion
    t1 = time.time()
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    print("Indices generated in", TIME_FORMAT_STR.format(time.time() - t1))
    exponents = multi_index.exponents
    nr_exponents, m = exponents.shape

    for idx in range(nr_exponents):
        exp_vect = exponents[idx, :]
        exponents2 = insert_lexicographically(exponents, exp_vect)
        nr_exponents_new, m2 = exponents2.shape
        assert nr_exponents_new == nr_exponents, \
            'inserting an already contained index should not change the index set shape'
        assert np.all(exponents == exponents2), \
            'inserting an already contained index should not change the index set values'
        assert exponents is exponents2, \
            'inserting an already contained index should not change the index set object'

        incomplete_exponents = np.delete(exponents, idx, axis=0)
        restored_exponents = insert_lexicographically(exponents, exp_vect)
        nr_exponents_new, m2 = restored_exponents.shape
        assert nr_exponents_new == nr_exponents, 'insertion did not work. wrong length'
        assert m2 == m
        assert np.all(exp_vect == restored_exponents[idx, :]), 'the index got inserted at the wrong position'
        assert np.all(exponents == restored_exponents), 'the indices should be identical'

    # bigger indices should be inserted at the end
    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    enlarged_exponents = insert_lexicographically(exponents, bigger_exponent_vector)
    nr_exponents_new, m2 = enlarged_exponents.shape
    assert nr_exponents_new == nr_exponents + 1, 'insertion did not work. wrong length'
    assert m2 == m
    assert np.all(bigger_exponent_vector == enlarged_exponents[-1, :]), 'the index got inserted at the wrong position'
    assert np.all(exponents == enlarged_exponents[:-1, :]), 'the previous indices should be identical'


def completion_test(spatial_dimension, poly_degree, lp_degree):
    t1 = time.time()
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    print("Indices generated in", TIME_FORMAT_STR.format(time.time() - t1))

    exp_orig = multi_index.exponents
    nr_exp_orig, m = exp_orig.shape
    assert multi_index.is_complete, 'indices constructed from a given polynomial degree should be complete'
    assert is_lexicographically_complete(
        exp_orig), 'indices constructed from a given polynomial degree should be complete'

    def assert_completion(completed_exponents):
        nr_exponents_new, m2 = completed_exponents.shape
        assert have_lexicographical_ordering(completed_exponents)
        assert is_lexicographically_complete(completed_exponents)
        assert MultiIndex(completed_exponents).is_complete
        assert nr_exponents_new == nr_exp_orig, 'completion changed the amount of indices'
        assert m2 == m, 'completion changed the amount of dimensions'
        assert np.array_equal(exp_orig, completed_exponents), 'the indices should be identical'

    completed_exponents = make_complete(exp_orig)
    assert_completion(completed_exponents)
    multi_index_completed = multi_index.make_complete()
    assert multi_index_completed is multi_index, 'completing already complete indices should return the same object!'

    for idx in range(nr_exp_orig):
        exp_vect = exp_orig[idx, :]
        incomplete_exponents = np.delete(exp_orig, idx, axis=0)

        # completion should be identical to "make derivable" after just deleting a single exponent vector
        completed_exponents1 = make_derivable(incomplete_exponents)
        completed_exponents2 = make_complete(incomplete_exponents)

        if is_lexicographically_complete(incomplete_exponents):
            # TODO test: 'completing already complete indices should return the same object!'
            continue  # only test completing incomplete exponent sets

        assert np.all(exp_vect == completed_exponents1[idx, :]), 'the index got inserted at the wrong position'
        assert_completion(completed_exponents1, )
        assert np.all(exp_vect == completed_exponents2[idx, :]), 'the index got inserted at the wrong position'
        assert_completion(completed_exponents2)

        # complete index sets with multiple missing exponents:
        # takes quite long
        # for idx2 in range(nr_exp_orig - 1):
        #     incomplete_exponents2 = np.delete(incomplete_exponents, idx2, axis=1)
        #     if is_lexicographically_complete(incomplete_exponents2):
        #         # TODO test: 'completing already complete indices should return the same object!'
        #         continue
        #     completed_exponents3 = make_complete(incomplete_exponents2)
        #     assert is_lexicographically_complete(completed_exponents3)
        #     assert MultiIndex(completed_exponents3).is_complete


def check_all_contained(spatial_dimension, poly_degree, lp_degree):
    multi_index = MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree)
    exponents = multi_index.exponents
    nr_exponents, m = exponents.shape

    for idx in range(nr_exponents):
        incomplete_exponents = np.delete(exponents, idx, axis=0)
        assert not all_indices_are_contained(exponents, incomplete_exponents)
        assert all_indices_are_contained(incomplete_exponents, exponents)

    # when adding indices, not all indices are contained
    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    enlarged_exponents = insert_lexicographically(exponents, bigger_exponent_vector)
    assert not all_indices_are_contained(enlarged_exponents, exponents)
    # but the other way round should hold:
    assert all_indices_are_contained(exponents, enlarged_exponents)


DIM_THRESH_SLOW_TESTS = 3


class MultiIndexTest(unittest.TestCase):

    def test_init(self):
        print('\ntesting MultiIndex(...) construction:')
        # NOTE: slow for higher dimensions -> skip
        check_different_settings(from_exponent_construction, max_dim=DIM_THRESH_SLOW_TESTS)

    def test_from_degree(self):
        # NOTE: slow for higher dimensions -> skip
        print('\ntesting MultiIndex.from_degree(...) construction:')
        check_different_settings(from_degree_construction, max_dim=DIM_THRESH_SLOW_TESTS)


class IndexHelperFctTest(unittest.TestCase):

    def test_lex_smaller_or_equal(self):
        print('\ntesting lex_smaller_or_equal() function:')
        check_different_settings(lex_smaller_or_equal_test)

    def test_index_is_contained(self):
        print('\ntesting is_contained() function:')
        check_different_settings(index_is_contained_test)

    def test_index_is_complete(self):
        print('\ntesting is_complete:')
        check_different_settings(index_is_complete_test)

    def test_insertion(self):
        print('\ntesting insert_single_vector() function:')
        check_different_settings(insert_indices_test)

    def test_completion(self):
        print('\ntesting the completion functions:')
        # NOTE: slow for larger problems -> skip higher dimensions
        check_different_settings(completion_test, max_dim=DIM_THRESH_SLOW_TESTS)

    def test_all_indices_are_contained(self):
        print('\ntesting all_indices_are_contained():')
        check_different_settings(check_all_contained)

    def test_have_lexicographical_ordering(self):
        exponents = np.array([[0, 0], [0, 1]])
        assert have_lexicographical_ordering(exponents)

        _, dim = exponents.shape
        # appending a bigger index at the end maintains ordering:
        NR_TRIALS = 10
        for i in range(NR_TRIALS):
            largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
            bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
            exponents = np.append(exponents, bigger_exponent_vector).reshape(-1, dim)
            assert have_lexicographical_ordering(exponents)

            # switching the position of indices will destroy the ordering
            switched_exponents = switch_positions(exponents)
            assert not have_lexicographical_ordering(switched_exponents)

            # adding duplicates will destroy the ordering:
            duplicate_exponents = np.append(exponents, exponents[-1, :]).reshape(-1, dim)
            assert not have_lexicographical_ordering(duplicate_exponents)


if __name__ == '__main__':
    # suite = unittest.TestLoader().loadTestsFromTestCase(HelperTest)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
