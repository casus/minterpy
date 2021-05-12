"""
Module of the MultiIndex class
"""
from copy import deepcopy
from typing import Optional

import numpy as np

from minterpy.global_settings import ARRAY, INT_DTYPE
from minterpy.jit_compiled_utils import (
    all_indices_are_contained,
    have_lexicographical_ordering,
)
from minterpy.multi_index_utils import (
    _expand_dim,
    get_exponent_matrix,
    insert_lexicographically,
    is_lexicographically_complete,
    make_complete,
    verify_lp_deg,
)

__all__ = ["MultiIndex"]

from minterpy.verification import check_shape, check_values


# TODO implement (set) comparison operations based on the multi index utils (>=, == ...)
class MultiIndex:
    def __init__(self, exponents: ARRAY, lp_degree=None, poly_deg_dtype=None):
        exponents = np.require(exponents, dtype=INT_DTYPE)
        check_shape(exponents, dimensionality=2)
        if exponents.shape[1] == 0:
            raise ValueError(
                f"the dimensionality of the given exponents is 0. shape: {exponents.shape}"
            )
        if not have_lexicographical_ordering(exponents):
            raise ValueError(
                "the multi_indices must be ordered lexicographically from last to first column"
            )
        self._exponents: ARRAY = exponents

        # TODO compute properly, max norm of the exponents?
        # while _get_poly_degree(exponents, __lp_degree) > poly_degree: ...
        self._lp_degree = verify_lp_deg(lp_degree)

        # TODO the polynomial degree can only be interpreted as integer?!
        # if poly_deg_dtype is None:
        #     self.poly_deg_dtype = float
        # else:
        #     self.poly_deg_dtype = poly_deg_dtype
        self.poly_deg_dtype = INT_DTYPE

        # self.poly_degree = _get_poly_degree(exponents, self._lp_degree)
        # if self.poly_degree == 0:
        #     raise ValueError('the degree must be bigger than 0')
        # if self.poly_degree % 1.0 == 0.0:
        # self.poly_degree = int(self.poly_degree)
        self.poly_degree = np.max(exponents)

        self._is_complete: Optional[bool] = None
        # for avoiding to complete the exponents multiple times
        self._exponents_completed: Optional[ARRAY] = None

    @classmethod
    def from_degree(cls, spatial_dimension: int, poly_degree: int, lp_degree=None):
        if type(poly_degree) is not int:
            raise TypeError("only integer polynomial degrees are supported.")
        if type(spatial_dimension) is not int:
            raise TypeError("spatial dimension must be given as integer.")
        lp_degree = verify_lp_deg(lp_degree)
        exponents = get_exponent_matrix(spatial_dimension, poly_degree, lp_degree)
        return cls(exponents, lp_degree=lp_degree, poly_deg_dtype=int)

    def expand_dim(self, dim):
        # TODO avoid transpose
        # print("self.__exponents",self.__exponents)
        # print("self.__exponents.T",self.__exponents.T)
        # print("_expand_dim(self.__exponents.T,dim)",_expand_dim(self.__exponents.T,dim))
        self._exponents = _expand_dim(self._exponents, dim)

    @property
    def exponents(
        self,
    ):  # read only: must not be altered since transformation matrices etc. depends on this
        return self._exponents

    @property
    def exponents_completed(self):
        # NOTE: exponents which are not complete ("with holes") cause problems with DDS, evaluation...
        # compute and store a completed version of the indices for these operations!
        if self._exponents_completed is None:  # lazy evaluation
            if self.is_complete:
                self._exponents_completed = self._exponents
            else:
                self._exponents_completed = make_complete(self._exponents)
        return self._exponents_completed

    @property
    def is_complete(self):
        # NOTE: exponents which are not complete ("with holes") cause problems with DDS, evaluation...
        if self._is_complete is None:  # lazy evaluation
            self._is_complete = is_lexicographically_complete(self.exponents)
            if self._is_complete:
                self._exponents_completed = self._exponents
        return self._is_complete

    @property
    def lp_degree(self):
        return self._lp_degree

    @lp_degree.setter
    def lp_degree(self, lp_degree):
        # TODO is a setter really meaningful for this attribute?
        #  should rather be computed from the exponents and be fixed afterwards
        if lp_degree <= 0.0:
            raise ValueError(
                f"The lp_degree needs to be a positive value! <{lp_degree}> given."
            )
        self._lp_degree = lp_degree

    @property
    def spatial_dimension(self):
        return self._exponents.shape[1]

    def __str__(self):
        return "\n".join(["MultiIndex", str(self._exponents)])

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        # returns the number of multi_indices
        return self._exponents.shape[0]

    def union(self):
        raise NotImplementedError("MultiIndex.union() is not implemented yet.")

    def __add__(self):
        raise NotImplementedError("MultiIndex.__add__() is not implemented yet.")

    def ordering(self, order):
        raise NotImplementedError("MultiIndex.ordering() is not implemented yet.")

    # def complete(self, order):
    #     raise NotImplementedError("MultiIndex.complete() is not implemented yet.")

    def copy_private_attributes(self, new_instance):
        new_instance._is_complete = self._is_complete
        new_instance._exponents_completed = self._exponents_completed

    # copying
    def __copy__(self):
        # TODO also copy tree if available
        # TODO maybe use always deepcopy since exponents will be always nested?!
        new_instance = self.__class__(
            self._exponents, self._lp_degree, self.poly_deg_dtype
        )
        self.copy_private_attributes(new_instance)
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = self.__class__(
            deepcopy(self._exponents),
            deepcopy(self._lp_degree),
            deepcopy(self.poly_deg_dtype),
        )
        self.copy_private_attributes(new_instance)
        return new_instance

    def contains_these_exponents(self, vectors: ARRAY) -> bool:
        return all_indices_are_contained(vectors, self._exponents)

    def is_sub_index_set_of(self, super_set: "MultiIndex") -> bool:
        return super_set.contains_these_exponents(self.exponents)

    def is_super_index_set_of(self, sub_set: "MultiIndex") -> bool:
        return self.contains_these_exponents(sub_set.exponents)

    def _new_instance_if_necessary(self, new_exponents: ARRAY) -> "MultiIndex":
        """constructs a new instance only if the exponents are different"""
        old_exponents = self._exponents
        if new_exponents is old_exponents:
            return self
        new_instance = self.__class__(new_exponents)
        # TODO add to the completed exponents and re-complete again
        if self._exponents_completed is not None:
            _exponents_completed = insert_lexicographically(
                self._exponents_completed, new_exponents
            )
            if (
                _exponents_completed is not self._exponents_completed
            ):  # some exponents have been added:
                # make complete again!
                _exponents_completed = make_complete(_exponents_completed)
            new_instance._exponents_completed = _exponents_completed
            # also set the exponents of the new_instance
            # TODO avoid redundancy. why store both _exponents and _exponents_completed?
            # new_instance._exponents = _exponents_completed
        return new_instance

    def add_exponents(self, exponents: ARRAY) -> "MultiIndex":
        exponents = np.require(exponents, dtype=INT_DTYPE)
        check_values(exponents)
        exponents = exponents.reshape(
            -1, self.spatial_dimension
        )  # convert input to 2D in expected shape
        #  NOTE: the insertion is expected to return the same identical array instance
        #  if all exponents are already contained!
        # -> no need to check for inclusion first!
        new_exponents = insert_lexicographically(self._exponents, exponents)
        return self._new_instance_if_necessary(new_exponents)

    def make_complete(self) -> "MultiIndex":
        # ATTENTION: the make_complete() fct should not be called with already complete idx sets (inefficient!)
        # should return the same object if already complete (avoid creation of identical instance)
        if self.is_complete:
            return self
        new_exponents = self.exponents_completed  # compute if necessary!
        new_instance = self.__class__(new_exponents)  # re-compute degree etc.
        # NOTE: avoid checking for completeness again!
        new_instance._is_complete = True
        new_instance._exponents_completed = new_exponents
        return new_instance

    # TODO make_derivable(): add (only) partial derivative exponent vectors,
    #  NOTE: not meaningful since derivation requires complete index sets anyway?
