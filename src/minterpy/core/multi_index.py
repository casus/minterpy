"""
Module of the MultiIndexSet class
"""
from copy import copy, deepcopy
from typing import Optional

import numpy as np

from minterpy.global_settings import ARRAY, INT_DTYPE, DEFAULT_LP_DEG
from minterpy.jit_compiled_utils import (
    all_indices_are_contained,
)
from minterpy.core.utils import (
    expand_dim,
    get_poly_degree,
    get_exponent_matrix,
    insert_lexicographically,
    is_complete,
    is_downward_closed,
    lex_sort,
    make_complete,
    make_downward_closed,
    multiply_indices,
)

from .verification import (
    check_shape,
    check_values,
    verify_lp_degree,
    verify_spatial_dimension,
)

__all__ = ["MultiIndexSet"]


# TODO implement (set) comparison operations based on the multi index utils (>=, == ...)
class MultiIndexSet:
    """A class to represent the set of multi-indices.

    The instances of this class provide the data structure for the exponents
    of multi-dimensional polynomials independently of the chosen basis in
    the polynomial space.

    Parameters
    ----------
    exponents : :class:`numpy:numpy.ndarray`
        Set of exponents given as a two-dimensional non-negative integer array
        of shape ``(N, m)``, where ``N`` is the number of multi-index elements
        (i.e., exponents) and ``m`` is the number of spatial dimension.
        Each row of the array indicates the vector of exponents.
    lp_degree : float
        :math:`p` of the :math:`l_p`-norm (i.e., the :math:`l_p`-degree)
        that is used to define the multi-index set.

    Notes
    -----
    - If the set of exponents is not given as an integer array, it is converted
      to an integer array once the instance has been constructed.
    - The resulting polynomial degree corresponds to the minimum degree
      that would include the given set of exponents with respect to
      the :math:`l_p`-degree.
    """

    def __init__(self, exponents: np.ndarray, lp_degree: float):

        # Check and assign the exponents
        exponents = np.require(exponents, dtype=INT_DTYPE)
        check_shape(exponents, dimensionality=2)

        # Keep only unique entries and sort lexicographically
        self._exponents = lex_sort(exponents)

        # Verify the given lp_degree
        self._lp_degree = verify_lp_degree(lp_degree)

        # Compute the polynomial degree given the exponents and lp-degree
        self._poly_degree = get_poly_degree(exponents, self._lp_degree)

        # Lazy evaluations (evaluated when accessed)
        self._is_complete: Optional[bool] = None
        self._is_downward_closed: Optional[bool] = None
        # for avoiding to complete the exponents multiple times
        # TODO: Possibly remove this property
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
        spatial_dimension = verify_spatial_dimension(spatial_dimension)
        lp_degree = verify_lp_degree(lp_degree)
        exponents = get_exponent_matrix(
            spatial_dimension, poly_degree, lp_degree
        )

        return cls(exponents, lp_degree=lp_degree)

    @property
    def exponents(self) -> np.ndarray:
        """Array of exponents in the form of multi-indices.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            A two-dimensional integer array of exponents in the form of
            lexicographically sorted (ordered) multi-indices.
            This is a read-only property and defined at construction.

        Notes
        -----
        - Each row corresponds to the exponent of a multi-dimensional
          polynomial basis while, each column corresponds to the exponent
          of a given dimension of a multi-dimensional polynomial basis.
        """
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
    def lp_degree(self) -> float:
        """:math:`p` of :math:`l_p`-norm used to define the multi-index set.

        Returns
        -------
        float
            The :math:`p` of the :math:`l_p`-norm
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
    def spatial_dimension(self) -> int:
        """The dimension of the domain space.

        Returns
        -------
        int
            The dimension of the domain space, on which a polynomial described
            by this instance of :py:class:`.MultiIndexSet` lives.
            It is equal to the number of elements (columns) in each element
            of multi-indices.
        """
        return self._exponents.shape[1]

    @property
    def is_complete(self) -> bool:
        """Return ``True`` if the ``exponents`` array is complete.

        Returns
        -------
        bool
            ``True`` if the ``exponents`` array is complete and ``False``
            otherwise.

        Notes
        -----
        - For a definition of completeness, refer to the relevant
          :doc:`section </how-to/multi-index-set/multi-index-set-complete>`
          of the Minterpy Documentation.
        """
        if self._is_complete is None:
            # Lazy evaluation
            self._is_complete = is_complete(
                self.exponents, self.poly_degree, self.lp_degree
            )

        return self._is_complete

    @property
    def is_downward_closed(self):
        """Return ``True`` if the ``exponents`` array is downward-closed.

        Returns
        -------
        bool
            ``True`` if the ``exponents`` array is downward-closed and
            ``False`` otherwise.

        Notes
        -----
        - Many Minterpy routines like DDS, Newton-evaluation, etc. strictly
          require a downward-closed exponent array (i.e., an array without
          lexicographical "holes") to work.
        - Refer to the Fundamentals section of the Minterpy Documentation
          for the definition of
          :doc:`multi-index sets </fundamentals/polynomial-bases>`.
        """
        if self._is_downward_closed is None:
            # Lazy evaluation
            self._is_downward_closed = is_downward_closed(self.exponents)

        return self._is_downward_closed

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
        exponents: np.ndarray,
        inplace=False,
    ) -> Optional["MultiIndexSet"]:
        """Add a set of exponents lexicographically into this instance.

        Parameters
        ----------
        exponents : `numpy:numpy.ndarray`
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

        # --- Identical exponents after addition
        if new_exponents is self._exponents:
            # Identical array
            if inplace:
                return None
            else:
                # Return a shallow copy of the current instance
                return copy(self)

        # --- Updated exponents after addition
        if inplace:
            self._exponents = new_exponents
            # The polynomial degree must be re-computed
            self._poly_degree = get_poly_degree(
                exponents=new_exponents, lp_degree=self.lp_degree
            )
            # Can't guarantee the updated exponents are still complete
            self._is_complete = None
            # nor downward-closed
            self._is_downward_closed = None
        else:
            # Create a new instance
            new_instance = self.__class__(
                exponents=new_exponents, lp_degree=self.lp_degree
            )

            return new_instance

    def expand_dim(
        self,
        new_dimension: int,
        inplace: bool = False,
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

        Raises
        ------
        ValueError
            If the new dimension is smaller than the current spatial
            dimension of the :py:class:`.MultiIndexSet` instance.

        Notes
        -----
        - If the new dimension is the same as the current spatial dimension
          of the :py:class:`.MultiIndexSet` instance, setting ``inplace``
          to ``False`` (the default) creates a shallow copy.
        """
        # Expand the dimension of the current exponents, i.e., add a new column
        expanded_exponents = expand_dim(
            self._exponents, new_dimension
        )

        # --- Identical exponents after expansion
        if expanded_exponents is self._exponents:
            if inplace:
                return None
            else:
                # Return a shallow copy of the current instance
                return copy(self)

        # --- Updated exponents after expansion
        if inplace:
            self._exponents = expanded_exponents
            # NOTE: Reset properties (if the exponent is only 0's,
            #       it remains complete and downward-closed;
            #       otherwise, no. None to be safe)
            self._is_complete = None
            self._is_downward_closed = None

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
        # --- Exponents already complete
        if self.is_complete:
            if inplace:
                return None

            # Otherwise, create a shallow copy of the current instance
            return copy(self)

        # --- Exponents made complete
        completed_exponents = make_complete(self.exponents, self.lp_degree)
        if inplace:
            # Modify the current instance
            self._exponents = completed_exponents
            # By construction, the current instance is now complete
            self._is_complete = True
            # A complete set is a downward-closed set
            self._is_downward_closed = True
        else:
            # Create a new instance
            new_instance = self.__class__(
                exponents=completed_exponents, lp_degree=self.lp_degree
            )
            # By construction, the new instance is complete
            new_instance._is_complete = True
            # A complete set is a downward-closed set
            new_instance._is_downward_closed = True

            return new_instance

    def make_downward_closed(
        self,
        inplace: bool = False,
    ) -> Optional["MultiIndexSet"]:
        """Create a downward-closed multi-index set from the current exponents.

        Parameters
        ----------
        inplace : bool, optional
            Flag to determine whether the current instance is modified
            in-place with the downward-closed set of exponents.
            If ``inplace`` is ``False``, a new :py:class:`.MultiIndexSet`
            instance is created.
            The default is ``False``.

        Returns
        -------
        `MultiIndexSet`, optional
            The multi-index set with the downward-closed set of exponents.
            If ``inplace`` is set to ``True``, then the modification
            is carried out in-place without an explicit output.

        Notes
        -----
        - If the current :py:class:`.MultiIndexSet` exponents are already
          complete, setting ``inplace`` to ``False`` (the default) creates
          a shallow copy.
        """
        # --- Exponents are already downward-closed
        if self.is_downward_closed:
            if inplace:
                return None

            # Otherwise, create a shallow copy of the current instance
            return copy(self)

        # --- Exponents made downward-closed
        downward_closed_exponents = make_downward_closed(self.exponents)
        if inplace:
            # Modify the current instance
            self._exponents = downward_closed_exponents
            # By construction, the current instance is now downward-closed
            self._is_downward_closed = True
            # ...but its completeness can't be guaranteed
            self._is_complete = None
        else:
            # Create a new instance
            new_instance = self.__class__(
                exponents=downward_closed_exponents, lp_degree=self.lp_degree
            )
            # By construction, the new instance is downward-closed
            new_instance._is_downward_closed = True

            return new_instance

    def __mul__(self, other: "MultiIndexSet") -> "MultiIndexSet":
        """Multiply an instance of MultiIndexSet with another.

        Parameters
        ----------
        other : MultiIndexSet
            The second operand of the multi-index set multiplication.

        Returns
        -------
        MultiIndexSet
            The product of two multi-index sets. If the :math:`l_p`-degrees
            of the operands are different, then the :math:`l_p`-degree of the
            product is the maximum of the two operands.
        """
        # Get the exponents of the operands
        exp_self = self.exponents
        exp_other = other.exponents

        # Decide the lp-degree of the product
        lp_degree_self = self.lp_degree
        lp_degree_other = other.lp_degree
        lp_degree_prod = max([lp_degree_self, lp_degree_other])

        # Take the product of the exponents (multi-indices multiplication)
        exp_prod = multiply_indices(exp_self, exp_other)

        return self.__class__(
            exponents=exp_prod, lp_degree=lp_degree_prod,
        )

    # TODO make_derivable(): add (only) partial derivative exponent vectors,
    # NOTE: not meaningful since derivation requires complete index sets anyway?
