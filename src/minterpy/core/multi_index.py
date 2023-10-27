"""
Module of the MultiIndexSet class
"""
from copy import copy, deepcopy
from typing import Optional

import numpy as np

from minterpy.global_settings import ARRAY, INT_DTYPE, DEFAULT_LP_DEG
from minterpy.jit_compiled_utils import (
    all_indices_are_contained,
    have_lexicographical_ordering,
)
from minterpy.core.utils import (
    expand_dim,
    get_poly_degree,
    get_exponent_matrix,
    insert_lexicographically,
    is_lexicographically_complete,
    make_complete,
)

from .verification import check_shape, check_values, verify_lp_degree

__all__ = ["MultiIndexSet"]


# TODO implement (set) comparison operations based on the multi index utils (>=, == ...)
class MultiIndexSet:
    """Class representation of the set of multi indices for a polynomial.

    The instances of this class provide storrage for the exponents of a multi variate
    polynomial independently of the base assumed in the polynomial space. Only
    the polynomial degree (w.r.t. a given `l_p` norm) as well as the dimension of the space are used to build the set of exponents.
    """

    def __init__(self, exponents: ARRAY, lp_degree: float):

        # Check and assign the exponents
        exponents = np.require(exponents, dtype=INT_DTYPE)
        check_shape(exponents, dimensionality=2)
        if not have_lexicographical_ordering(exponents):
            raise ValueError(
                "The multi-index set must be lexicographically ordered from "
                "the last to the first column."
            )
        self._exponents: ARRAY = exponents

        # Verify the given lp_degree
        self._lp_degree = verify_lp_degree(lp_degree)

        # Compute the polynomial degree given the exponents and lp-degree
        self._poly_degree = get_poly_degree(exponents, self._lp_degree)

        self._is_complete: Optional[bool] = None
        # for avoiding to complete the exponents multiple times
        self._exponents_completed: Optional[ARRAY] = None

    @classmethod
    def from_degree(
        cls,
        spatial_dimension: int,
        poly_degree: int,
        lp_degree: float = DEFAULT_LP_DEG,
    ):
        """Create from given spatial dimension, poly. degree, and lp-degree."""
        if type(poly_degree) is not int:
            raise TypeError("only integer polynomial degrees are supported.")
        if type(spatial_dimension) is not int:
            raise TypeError("spatial dimension must be given as integer.")
        lp_degree = verify_lp_degree(lp_degree)
        exponents = get_exponent_matrix(
            spatial_dimension, poly_degree, lp_degree
        )

        return cls(exponents, lp_degree=lp_degree)

    @property
    def exponents(
        self,
    ):
        """Array of exponents.

        :return: A 2D :class:`ndarray` which stores the exponents, where the first axis refers to the explicit exponent and the second axis refers to the variable which is powerd by the entry at this array position. These exponents are always lexicographically ordered w.r.t. the first axis.

        :rtype: np.ndarray
        """
        # read only: must not be altered since transformation matrices etc. depends on this

        return self._exponents

    @property
    def exponents_completed(self):
        """
        The completed version of the exponent array.

        :return: A complete array of the ``exponents``, i.e. in lexicographical ordering, every exponent which is missing in ``exponents``, will be filled with the right entry.
        :rtype: np.ndarray

        """
        # NOTE: exponents which are not complete ("with holes") cause problems with DDS, evaluation...
        # compute and store a completed version of the indices for these operations!
        if self._exponents_completed is None:  # lazy evaluation
            if self.is_complete:
                self._exponents_completed = self._exponents
            else:
                self._exponents_completed = make_complete(
                    self._exponents, self.lp_degree
                )
        return self._exponents_completed

    @property
    def is_complete(self):
        """Returns :class:`True` if the ``exponent`` array is complete.

        :rtype: bool
        """
        # NOTE: exponents which are not complete ("with holes") cause problems with DDS, evaluation...
        if self._is_complete is None:  # lazy evaluation
            self._is_complete = is_lexicographically_complete(self.exponents)
            if self._is_complete:
                self._exponents_completed = self._exponents
        return self._is_complete

    @property
    def lp_degree(self) -> float:
        """:math:`l_p` of :math:`l_p`-norm used to define the multi-index set.

        Returns
        -------
        float
            The :math:`l_p` of the :math:`l_p`-norm
            (i.e., the :math:`l_p`-degree) that is used to define
            the multi-index set. This property is read-only
            and defined at construction.
        """
        return self._lp_degree

    @property
    def poly_degree(self) -> int:
        """The polynomial degree of the multi-index set.

        Returns
        -------
        int
            The polynomial degree of the multi-index set. This property is
            read-only and inferred from a given multi-index set and lp-degree.
        """
        return self._poly_degree

    @property
    def spatial_dimension(self):
        """The dimentsion of the domain space.

         :return: The dimension of the space, where a polynomial described by this ``multi_index`` lives on. It is equal to the number of powers in each exponent vector.

        :rtype: int

        """
        return self._exponents.shape[1]

    def __str__(self):
        return "\n".join(["MultiIndexSet", str(self._exponents)])

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        # returns the number of multi_indices
        return self._exponents.shape[0]

    def __add__(self):
        """This function is not implemented yet.

        :raise NotImplementedError: If this function is called.

        """
        raise NotImplementedError("MultiIndexSet.__add__() is not implemented yet.")

    def union(self):
        """This function is not implemented yet.

        :raise NotImplementedError: If this function is called.

        """
        raise NotImplementedError("MultiIndexSet.union() is not implemented yet.")

    def ordering(self, order):
        """This function is not implemented yet.

        :raise NotImplementedError: If this function is called.

        """
        raise NotImplementedError("MultiIndexSet.ordering() is not implemented yet.")

    # def complete(self, order):
    #     raise NotImplementedError("MultiIndexSet.complete() is not implemented yet.")

    def copy_private_attributes(self, new_instance):
        """Copys the private attributes to another instance of :class:`MultiIndexSet`

        .. todo::
            - use the ``__copy__`` hook instead!

        """
        new_instance._is_complete = self._is_complete
        new_instance._exponents_completed = self._exponents_completed

    # copying
    def __copy__(self):
        """Creates of a shallow copy.

        This function is called, if one uses the top-level function ``copy()`` on an instance of this class.

        :return: The copy of the current instance.
        :rtype: MultiIndexSet

        See Also
        --------
        copy.copy
            copy operator form the python standard library.
        """
        # TODO also copy tree if available
        # TODO maybe use always deepcopy since exponents will be always nested?!
        new_instance = self.__class__(
            self._exponents, self._lp_degree
        )
        self.copy_private_attributes(new_instance)

        return new_instance

    def __deepcopy__(self, memo):
        """Creates of a deepcopy.

        This function is called, if one uses the top-level function ``deepcopy()`` on an instance of this class.

        :return: The deepcopy of the current instance.
        :rtype: MultiIndexSet

        See Also
        --------
        copy.deepcopy
            copy operator form the python standard library.
        """
        new_instance = self.__class__(
            deepcopy(self.exponents),
            deepcopy(self._lp_degree),
        )
        self.copy_private_attributes(new_instance)

        return new_instance

    def contains_these_exponents(self, vectors: ARRAY) -> bool:
        """Checks if this instance contains a given set of exponents.

        :param vectors: Exponents to be checked.
        :type vectors: np.ndarray

        """
        return all_indices_are_contained(vectors, self._exponents)

    def is_sub_index_set_of(self, super_set: "MultiIndexSet") -> bool:
        """Checks if this instance is a subset of the given instance of :class:`MultiIndexSet`.

        :param super_set: Superset to be checked.
        :type super_set: MultiIndexSet

        .. todo::
            - use comparison hooks for this functionality
        """
        return super_set.contains_these_exponents(self.exponents)

    def is_super_index_set_of(self, sub_set: "MultiIndexSet") -> bool:
        """Checks if this instance is a super of the given instance of :class:`MultiIndexSet`.

        :param sub_set: Subset to be checked.
        :type super_set: MultiIndexSet

        .. todo::
            - use comparison hooks for this functionality
        """
        return self.contains_these_exponents(sub_set.exponents)

    def _new_instance_if_necessary(self, new_exponents: ARRAY) -> "MultiIndexSet":
        """constructs a new instance only if the exponents are different

        .. todo::
            - this should be a separate function rather than a member function.
            - this should use the comparison hooks instead of attribute checking.
        """
        old_exponents = self._exponents
        if new_exponents is old_exponents:
            return self
        new_instance = self.__class__(new_exponents, self.lp_degree)
        # TODO add to the completed exponents and re-complete again
        if self._exponents_completed is not None:
            _exponents_completed = insert_lexicographically(
                self._exponents_completed, new_exponents
            )
            if (
                _exponents_completed is not self._exponents_completed
            ):  # some exponents have been added:
                # make complete again!
                _exponents_completed = make_complete(
                    _exponents_completed, self.lp_degree
                )
            new_instance._exponents_completed = _exponents_completed
            # also set the exponents of the new_instance
            # TODO avoid redundancy. why store both _exponents and _exponents_completed?
            # new_instance._exponents = _exponents_completed
        return new_instance

    def add_exponents(
        self,
        exponents: ARRAY,
        inplace=False
    ) -> Optional["MultiIndexSet"]:
        """Add a set of exponents lexicographically into this instance.

        Parameters
        ----------
        exponents : ARRAY
            Array of exponents to be added. The set of exponents must be of
            integer type.
        inplace : bool, optional
            Flag to determine whether the current instance is modified
            in-place with the complete exponents. If ``inplace`` is ``False``,
            a new :py:class:`.MultiIndexSet` instance is created.
            The default is ``False``.

        Returns
        -------
        `MultiIndexSet`, optional
            The multi-index set with an updated set of exponents.
            If ``inplace`` is set to ``True``, then the modification
            is carried out in-place without an explicit output
            (it returns ``None``).

        Notes
        -----
        - The array of added exponents must be of integer types; non-integer
          types will be converted to integer.
        - If the current :py:class:`.MultiIndexSet` exponents already
          include the added set of exponents and ``inplace`` is set
          to ``False`` (the default) a shallow copy is created.
        """
        # --- Pre-process the input exponents
        exponents = np.require(exponents, dtype=INT_DTYPE)
        check_values(exponents)
        # convert input to 2D in expected shape
        exponents = exponents.reshape(-1, self.spatial_dimension)

        #  NOTE: If all added exponents are already contained, identical array
        #        is returned.
        new_exponents = insert_lexicographically(self._exponents, exponents)
        if new_exponents is self._exponents:
            # Identical array
            if inplace:
                return None
            else:
                # Return a shallow copy of the current instance
                return copy(self)
        else:
            # The exponents have been updated
            if inplace:
                self._exponents = new_exponents
                # The polynomial degree must be re-computed
                self._poly_degree = get_poly_degree(
                    exponents=new_exponents, lp_degree=self.lp_degree
                )
                # Can't guarantee the updated exponents are still complete
                self._is_complete = None
            else:
                # Create a new instance
                new_instance = self.__class__(
                    exponents=new_exponents, lp_degree=self.lp_degree
                )

                return new_instance

    def expand_dim(
        self,
        new_dimension: int,
        inplace: bool = False
    ) -> Optional["MultiIndexSet"]:
        """Expand the dimension of the multi-index set.

        After expansion, the value of exponents in the new dimension is 0.

        Parameters
        ----------
        new_dimension : int
            The new spatial dimension. It must be larger than or equal
            to the current spatial dimension of the multi-index set.
        inplace : bool, optional
            Flag to determine whether the current instance is modified
            in-place with the expanded dimension. If ``inplace`` is ``False``,
            a new :py:class:`.MultiIndexSet` instance is created.
            The default is ``False``.

        Returns
        -------
        `MultiIndexSet`, optional
            The multi-index set with an expanded dimension.
            If ``inplace`` is set to ``True``, then the modification
            is carried out in-place without an explicit output.

        Notes
        -----
        - If the new dimension is the same as the current spatial dimension
          of the :py:class:`.MultiIndexSet` instance, setting ``inplace``
          to ``False`` (the default) creates a shallow copy.

        Raises
        ------
        ValueError
            If the target dimension is smaller than the current spatial
            dimension of the :py:class:`.MultiIndexSet` instance.
        """
        # Expand the dimension of the current exponents, i.e., add a new column
        expanded_exponents = expand_dim(
            self._exponents, new_dimension
        )

        if expanded_exponents is self._exponents:
            # Identical exponents after expansion
            if inplace:
                return None
            else:
                # Return a shallow copy of the current instance
                return copy(self)
        else:
            # The exponents have been updated
            if inplace:
                self._exponents = expanded_exponents
                # By construction, the exponents are not complete anymore
                self._is_complete = False
            else:
                # Create a new instance
                new_instance = self.__class__(
                    exponents=expanded_exponents, lp_degree=self.lp_degree
                )

                return new_instance

    def make_complete(
        self,
        inplace: bool = False,
    ) -> Optional["MultiIndexSet"]:
        """Create a complete multi-index set from the current exponents.

        Parameters
        ----------
        inplace : bool, optional
            Flag to determine whether the current instance is modified
            in-place with the complete exponents. If ``inplace`` is ``False``,
            a new :py:class:`.MultiIndexSet` instance is created.
            The default is ``False``.

        Returns
        -------
        `MultiIndexSet`, optional
            The multi-index set with a complete set of exponents.
            If ``inplace`` is set to ``True``, then the modification
            is carried out in-place without an explicit output.

        Notes
        -----
        - If the current :py:class:`.MultiIndexSet` exponents are already
          complete, setting ``inplace`` to ``False`` (the default) creates
          a shallow copy.
        """
        if self.is_complete:
            # Already complete before
            if inplace:
                return None
            else:
                # Create a shallow copy of the current instance
                return copy(self)
        else:
            completed_exponents = make_complete(self.exponents, self.lp_degree)
            if inplace:
                # Modify the current instance
                self._exponents = completed_exponents
                # By construction, the current instance is now complete
                self._is_complete = True
            else:
                # Create a new instance
                new_instance = self.__class__(
                    exponents=completed_exponents, lp_degree=self.lp_degree
                )
                # By construction, the new instance is complete
                new_instance._is_complete = True

                return new_instance

    # TODO make_derivable(): add (only) partial derivative exponent vectors,
    # NOTE: not meaningful since derivation requires complete index sets anyway?
