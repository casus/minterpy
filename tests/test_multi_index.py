"""
Test suite for multi_index.py
"""
import pytest

from conftest import SpatialDimension, PolyDegree, LpDegree,assert_call
from numpy.testing import assert_,assert_equal
import numpy as np
from minterpy import MultiIndex
from minterpy.jit_compiled_utils import (all_indices_are_contained,
                                         have_lexicographical_ordering,
                                         index_is_contained,
                                         lex_smaller_or_equal)
from minterpy.multi_index_utils import _gen_multi_index_exponents,_get_poly_degree,find_match_between


MIN_POLY_DEG = 1
MAX_POLY_DEG = 5
SEED = 12345678

def get_lex_bigger(index, bigger_by_1=False,seed = None):
    if seed is None:
        seed = SEED
    out = index.copy()
    m = len(index)
    if m == 1:
        rnd_dim = 0
    else:
        rnd_dim = np.random.randint(0, m - 1)
    out[rnd_dim] += 1
    if not bigger_by_1:
        out[
            :rnd_dim
        ] = 0  # setting all previous entries to 0 does not change the lexicographical ordering
    return out



# fixtures for number of monomials
number_of_monomials = [1,2]

@pytest.fixture(params = number_of_monomials)
def NumOfMonomials(request):
    return request.param


def build_rnd_exponents(dim,n,seed = None):
    """Build random exponents.

    For later use, if ``MultiIndex`` will accept arbitrary exponents again.

    :param dim: spatial dimension
    :param n: number of random monomials

    Notes
    -----
    Exponents are generated within the intervall ``[MIN_POLY_DEG,MAX_POLY_DEG]``

    """
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    return np.random.randint(MIN_POLY_DEG,MAX_POLY_DEG,(n,dim))


# test initialization
def test_init_from_exponents(SpatialDimension,PolyDegree,LpDegree):
    exponents = _gen_multi_index_exponents(SpatialDimension,PolyDegree,LpDegree)
    assert_call(MultiIndex,exponents)
    assert_call(MultiIndex,exponents, lp_degree = LpDegree)
    multi_index = MultiIndex(exponents, lp_degree = LpDegree)
    assert_(isinstance(multi_index, MultiIndex))
    assert_equal(exponents, multi_index.exponents)
    assert_(multi_index.lp_degree == LpDegree)
    assert_(multi_index.poly_degree == PolyDegree)

    number_of_monomials,dim = exponents.shape
    assert_(len(multi_index) == number_of_monomials)
    assert_(multi_index.spatial_dimension == dim)


def test_init_from_degree(SpatialDimension,PolyDegree,LpDegree):
    assert_call(MultiIndex.from_degree,SpatialDimension,PolyDegree)
    assert_call(MultiIndex.from_degree,SpatialDimension,PolyDegree,lp_degree = LpDegree)
    multi_index = MultiIndex.from_degree(SpatialDimension,PolyDegree,LpDegree)

    assert_(isinstance(multi_index, MultiIndex))
    assert_(multi_index.lp_degree == LpDegree)
    assert_(multi_index.poly_degree == PolyDegree)

    exponents = multi_index.exponents
    number_of_monomials, dim = exponents.shape
    assert_(dim == SpatialDimension)
    assert_(np.min(exponents) == 0)
    assert_(np.max(exponents) == PolyDegree)
    #assert_(_get_poly_degree(exponents, LpDegree)==PolyDegree)
    assert_(np.max(exponents) == PolyDegree)

    exponents_recur = _gen_multi_index_exponents(SpatialDimension,PolyDegree,LpDegree)
    if exponents_recur.shape[0] > number_of_monomials:
        raise AssertionError()
    if exponents_recur.shape[0] == number_of_monomials:
        assert_equal(exponents, exponents_recur)
    else:
        match_positions = find_match_between(exponents_recur, exponents)
        selected_exponents = exponents[match_positions, :]
        assert_equal(selected_exponents, exponents_recur)



# test utilities

def test_index_is_contained(SpatialDimension,PolyDegree,LpDegree):
    multi_index = MultiIndex.from_degree(SpatialDimension,PolyDegree,LpDegree)
    exponents = multi_index.exponents
    number_of_monomials, dim = exponents.shape
    assert_(dim == SpatialDimension)
    for exponent_vector in exponents:
        assert_(index_is_contained(exponents, exponent_vector))

    largest_exponent_vector = exponents[-1, :]  # last / biggest exponent vector
    bigger_exponent_vector = get_lex_bigger(largest_exponent_vector)
    assert_(not index_is_contained(exponents, bigger_exponent_vector))

    for i,exponent_vector in enumerate(exponents):
        print(exponents)
        exponents2 = np.delete(exponents, i, axis=0)
        print(exponents2)
        print(exponents[i] in exponents2)
        assert_(not index_is_contained(exponents, exponent_vector))
