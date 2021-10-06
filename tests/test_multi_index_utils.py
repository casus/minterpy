"""
Test suite for multi_index_utils.py
"""
import pytest

from conftest import SpatialDimension, PolyDegree, LpDegree,assert_call,build_rnd_exponents
from numpy.testing import assert_,assert_equal
import numpy as np

from minterpy.jit_compiled_utils import (all_indices_are_contained,
                                         have_lexicographical_ordering,
                                         index_is_contained,
                                         lex_smaller_or_equal)
from minterpy.core.utils import _gen_multi_index_exponents,_get_poly_degree,find_match_between,get_exponent_matrix,is_lexicographically_complete,insert_lexicographically,is_lexicographically_complete,make_derivable,make_complete


MIN_POLY_DEG = 1
MAX_POLY_DEG = 5
SEED = 12345678

def get_random_dim(spatial_dimension,seed = None):
    if seed is None:
        seed = SEED
    if spatial_dimension == 1:
        rnd_dim = 0
    else:
        rnd_dim = np.random.randint(0, spatial_dimension - 1)
    return rnd_dim

def get_lex_bigger(index, bigger_by_1=False,seed = None):
    rnd_dim = get_random_dim(len(index))
    out = index.copy()
    out[rnd_dim] += 1
    # setting all previous entries to 0 does not change the lexicographical ordering
    if not bigger_by_1:
        out[:rnd_dim] = 0
    return out



# fixtures for number of monomials
number_of_monomials = [1,2]

@pytest.fixture(params = number_of_monomials)
def NumOfMonomials(request):
    return request.param


# test utilities

def test_index_is_contained(SpatialDimension,PolyDegree,LpDegree):
    exponents = get_exponent_matrix(SpatialDimension,PolyDegree,LpDegree)
    number_of_monomials, dim = exponents.shape
    assert_(dim == SpatialDimension)

    # TODO: is it necessary to do this for every element?
    for exponent_vector in exponents:
        assert_(index_is_contained(exponents, exponent_vector))

    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    assert_(not index_is_contained(exponents, bigger_exponent_vector))

    deleted_exponent_vector = exponents[0]
    exponents2 = np.delete(exponents,0,axis = 0)
    assert_(not index_is_contained(exponents2, deleted_exponent_vector))


def test_lex_smaller_or_equal(SpatialDimension,PolyDegree,LpDegree):
    exponents = get_exponent_matrix(SpatialDimension,PolyDegree,LpDegree)
    number_of_monomials, dim = exponents.shape

    # TODO: is it necessary to do this for every element?
    for exp_vect in exponents:
        assert_(lex_smaller_or_equal(exp_vect, exp_vect))
        bigger_exponent_vector = get_lex_bigger(exp_vect)
        assert_(lex_smaller_or_equal(exp_vect, bigger_exponent_vector))
        assert_(not lex_smaller_or_equal(bigger_exponent_vector, exp_vect))


def test_is_lexicographically_complete(SpatialDimension,PolyDegree,LpDegree):
    exponents = get_exponent_matrix(SpatialDimension,PolyDegree,LpDegree)
    assert_(is_lexicographically_complete(exponents))

    # deleting the smallest vector destroys completeness
    assert_(not is_lexicographically_complete(
        np.delete(exponents, 0, axis=0)
        ))

    if PolyDegree!=1:
        assert_(not is_lexicographically_complete(
            np.delete(exponents, 1, axis=0)
            ))


    # deleting the biggest vector maintains completeness
    assert_(is_lexicographically_complete(
        np.delete(exponents, -1, axis=0)
        ))

    # TODO: shall not use functions which are not tested!
    #bigger_exponent_vector = expomemts[-1,:].copy()  # independent copy!
    #bigger_exponent_vector[0] += 2  # introduces "hole"
    #multi_index2 = multi_index.add_exponents(bigger_exponent_vector)
    #exponents2 = multi_index2.exponents
    #assert not is_lexicographically_complete(exponents2)
    #if multi_index2.is_complete:
    #assert not multi_index2.is_complete


def test_insert_indices(SpatialDimension,PolyDegree,LpDegree):
    exponents = get_exponent_matrix(SpatialDimension,PolyDegree,LpDegree)
    number_of_monomials, dim = exponents.shape

    # TODO: is it necessary to do this for every element?
    for idx, exp_vect in enumerate(exponents):

        exponents2 = insert_lexicographically(exponents, exp_vect)
        assert_equal(exponents2,exponents)

        incomplete_exponents = np.delete(exponents, idx, axis=0)
        restored_exponents = insert_lexicographically(exponents, exp_vect)
        assert_equal(restored_exponents,exponents)

    # bigger indices should be inserted at the end
    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    enlarged_exponents = insert_lexicographically(exponents, bigger_exponent_vector)
    assert_(enlarged_exponents.shape[0] == (number_of_monomials+1))
    assert_(enlarged_exponents.shape[1] == dim)
    assert_equal(enlarged_exponents[:-1],exponents)




def test_make_complete(SpatialDimension,PolyDegree,LpDegree):
    exponents = get_exponent_matrix(SpatialDimension,PolyDegree,LpDegree)
    assert_(is_lexicographically_complete(exponents))

    exp_vect = exponents[0]
    incomplete_exponents = np.delete(exponents, 0, axis=0)
    # completion should be identical to "make derivable" after just deleting a single exponent vector
    completed_exponents1 = make_derivable(incomplete_exponents)
    completed_exponents2 = make_complete(incomplete_exponents,LpDegree)

    assert(is_lexicographically_complete(completed_exponents1))
    assert(is_lexicographically_complete(completed_exponents2))

    assert_equal(exp_vect, completed_exponents1[0, :])
    assert_equal(exp_vect, completed_exponents2[0, :])

    if PolyDegree!=1:
        exp_vect = exponents[1]
        incomplete_exponents = np.delete(exponents, 1, axis=0)
        # completion should be identical to "make derivable" after just deleting a single exponent vector
        completed_exponents1 = make_derivable(incomplete_exponents)
        completed_exponents2 = make_complete(incomplete_exponents,LpDegree)

        assert(is_lexicographically_complete(completed_exponents1))
        assert(is_lexicographically_complete(completed_exponents2))

        assert_equal(exp_vect, completed_exponents1[1, :])
        assert_equal(exp_vect, completed_exponents2[1, :])


def test_all_indices_are_contained(SpatialDimension,PolyDegree,LpDegree):
    exponents = get_exponent_matrix(SpatialDimension,PolyDegree,LpDegree)

    # TODO: is it necessary to do this for every element?
    for idx in range(exponents.shape[0]):
        incomplete_exponents = np.delete(exponents, idx, axis=0)
        assert_(not all_indices_are_contained(exponents, incomplete_exponents))
        assert_(all_indices_are_contained(incomplete_exponents, exponents))

    # when adding indices, not all indices are contained <-- ???
    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    enlarged_exponents = insert_lexicographically(exponents, bigger_exponent_vector)
    assert_(not all_indices_are_contained(enlarged_exponents, exponents))
    # but the other way round should hold:
    assert_(all_indices_are_contained(exponents, enlarged_exponents))
