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
    union_indices,
)

from .verification import (
    check_dimensionality,
    check_values,
    verify_lp_degree,
    verify_spatial_dimension,
    verify_poly_degree,
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
        To create an empty set, an empty two-dimensional array may be passed.
    lp_degree : float
        :math:`p` of the :math:`l_p`-norm (i.e., the :math:`l_p`-degree)
        that is used to define the multi-index set. The value of
        ``lp_degree`` must be a positive float (:math:`p > 0`).

    Notes
    -----
    - If the set of exponents is not given as an integer array, it is converted
      to an integer array once the instance has been constructed.
    - The resulting polynomial degree corresponds to the minimum degree
      that would include the given set of exponents with respect to
      the :math:`l_p`-degree.
    """

    def __init__(self, exponents: np.ndarray, lp_degree: float):

        # Verify the given lp_degree
        self._lp_degree = verify_lp_degree(lp_degree)

        # Check exponent values
        # Must be non-negative
        check_values(exponents, negative=False)
        # Must be two-dimensional
        check_dimensionality(exponents, dimensionality=2)
        # Must be integer
        exponents = np.require(exponents, dtype=INT_DTYPE)

        # Empty set
        if exponents.size == 0:
            self._poly_degree = None
            if exponents.shape[1] == 0:
                self._exponents = np.empty((0, 0), dtype=INT_DTYPE)
            else:
                self._exponents = exponents
            self._is_complete = False
            self._is_downward_closed = False
            self._exponents_completed = None

            return

        # Keep only unique entries and sort lexicographically
        self._exponents = lex_sort(exponents)

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
    ) -> "MultiIndexSet":
        """Create an instance from given spatial dim., poly., and lp-degrees.

        Parameters
        ----------
        spatial_dimension : int
            Spatial dimension of the multi-index set (:math:`m`); the value of
            ``spatial_dimension`` must be a positive integer (:math:`m > 0`).
        poly_degree : int
            Polynomial degree of the multi-index set (:math:`n`); the value of
            ``poly_degree`` must be a non-negative integer (:math:`n \geq 0`).
        lp_degree : float, optional
            :math:`p` of the :math:`l_p`-norm (i.e., the :math:`l_p`-degree)
            that is used to define the multi-index set. The value of
            ``lp_degree`` must be a positive float (:math:`p > 0`).
            If not specified, ``lp_degree`` is assigned with the value of
            :math:`2.0`.

        Returns
        -------
        MultiIndexSet
            An instance of :py:class:`.MultiIndexSet` in the given
            ``spatial_dimension`` with a complete set of exponents with respect
            to the given ``poly_degree`` and ``lp_degree``.
        """
        # Verify the parameters
        poly_degree = verify_poly_degree(poly_degree)
        spatial_dimension = verify_spatial_dimension(spatial_dimension)
        lp_degree = verify_lp_degree(lp_degree)

        # Construct a complete set of multi-indices as the exponents
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
        """Return the cardinality of the multi-index set."""
        return self._exponents.shape[0]

    def __add__(self):
        """This function is not implemented yet.

        :raise NotImplementedError: If this function is called.

        """
        raise NotImplementedError("MultiIndexSet.__add__() is not implemented yet.")

    def __deepcopy__(self, memo):
        """Create of a deepcopy.

        This function is called, if one uses the top-level function
        ``deepcopy()`` on an instance of this class.

        Returns
        -------
        `MultiIndexSet`
            A deep copy of the current instance.

        Notes
        -----
        - Some properties of :class:`MultiIndexSet` are lazily evaluated,
          this custom implementation of `copy()` allows the values of
          already computed properties to be copied.
        - A deep copy contains a deep copy of the property exponents.

        See Also
        --------
        copy.deepcopy
            copy operator form the python standard library.
        """
        # Create a new empty instance
        exponent = np.empty(shape=(0, self.spatial_dimension), dtype=INT_DTYPE)
        new_instance = self.__class__(exponent, self._lp_degree)

        # Some properties are lazily evaluated, copy to avoid re-computation
        new_instance._exponents = self._exponents.copy()
        new_instance._lp_degree = self._lp_degree
        new_instance._poly_degree = self._poly_degree
        new_instance._is_complete = self._is_complete
        new_instance._is_downward_closed = self._is_downward_closed

        return new_instance

    def ordering(self, order):
        """This function is not implemented yet.

        :raise NotImplementedError: If this function is called.

        """
        raise NotImplementedError("MultiIndexSet.ordering() is not implemented yet.")

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
        if len(self) == 0:
            # The empty set is a subset of every set.
            return True
        if len(super_set) == 0:
            # Any set, except the empty set, is not a subset of the empty set
            return False

        return super_set.contains_these_exponents(self.exponents)

    def is_super_index_set_of(self, sub_set: "MultiIndexSet") -> bool:
        """Checks if this instance is a super of the given instance of :class:`MultiIndexSet`.

        :param sub_set: Subset to be checked.
        :type super_set: MultiIndexSet

        .. todo::
            - use comparison hooks for this functionality
        """
        if len(sub_set) == 0:
            # Every set is a superset of the empty set
            return True
        if len(self) == 0:
            # The empty set is not a superset of any set except the empty set
            return False

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
        # Check values, must be non-negative
        check_values(exponents, negative=False)
        # Must be integer
        exponents = np.require(exponents, dtype=INT_DTYPE)

        #  NOTE: If all added exponents are already contained,
        #        identical array is returned.
        if len(self) == 0:
            new_exponents = np.atleast_2d(exponents)
            check_dimensionality(new_exponents, dimensionality=2)
        else:
            # convert input to 2D in expected shape
            exponents = exponents.reshape(-1, self.spatial_dimension)
            new_exponents = insert_lexicographically(
                self._exponents,
                exponents
            )

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
        # --- Empty set
        if len(self) == 0:
            raise ValueError(
                "An empty multi-index set cannot be made complete!"
            )

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
        # --- Empty set
        if len(self) == 0:
            raise ValueError(
                "An empty multi-index set cannot be made downward-closed!"
            )

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

    def union(
        self,
        other: "MultiIndexSet",
        inplace: bool = False
    ) -> Optional["MultiIndexSet"]:
        """Take the union between two MultiIndexSet instances.

        Parameters
        ----------
        other : `MultiIndexSet`
            The second operand of the multi-index set union (i.e., ``other`` in
            ``self | other``).
        inplace : bool, optional
            Flag to determine whether the current instance is modified
            in-place with the union of the two multi-indices.
            If ``inplace`` is ``False``, a new :py:class:`.MultiIndexSet`
            instance is created.
            The default is ``False``.

        Returns
        -------
        `MultiIndexSet`, optional
            The multi-index set having the union of the two sets of exponents.
            If ``inplace`` is set to ``True``, then the modification
            is carried out in-place without an explicit output.

        Notes
        -----
        - If the operands are equal in value, setting ``inplace`` to ``False``
          (the default) creates a shallow copy.
        """
        # If equal no need to make any union
        if self == other:
            if inplace:
                # Do nothing
                return None

            return copy(self)

        # Get the exponents of the operands
        exponents_self = self.exponents
        exponents_other = other.exponents

        # Take the union of the exponents
        exponents_union = union_indices(exponents_self, exponents_other)

        # Decide the lp-degree of the product
        lp_degree_union = max([self.lp_degree, other.lp_degree])

        if inplace:
            self._exponents = exponents_union
            self._lp_degree = lp_degree_union
            # The polynomial degree must be re-computed
            self._poly_degree = get_poly_degree(
                exponents=exponents_union, lp_degree=lp_degree_union
            )
            # Can't guarantee the updated exponents are still complete
            self._is_complete = None
            # nor downward-closed
            self._is_downward_closed = None
        else:
            return self.__class__(
                exponents=exponents_union, lp_degree=lp_degree_union
            )

    def multiply(
        self,
        other: "MultiIndexSet",
        inplace: bool = False,
    ) -> Optional["MultiIndexSet"]:
        """Multiply an instance of `MultiIndexSet` with another."""
        # Get the exponents of the operands
        exp_self = self.exponents
        exp_other = other.exponents

        # Take the product of the exponents (multi-indices multiplication)
        exp_prod = multiply_indices(exp_self, exp_other)

        # Decide the lp-degree of the product
        lp_degree_self = self.lp_degree
        lp_degree_other = other.lp_degree
        lp_degree_prod = max([lp_degree_self, lp_degree_other])

        if inplace:
            self._exponents = exp_prod
            self._lp_degree = lp_degree_prod
            # NOTE: Reset properties
            self._is_complete = None
            self._is_downward_closed = None

        else:
            return self.__class__(
                exponents=exp_prod, lp_degree=lp_degree_prod,
            )

    def __eq__(self, other: "MultiIndexSet") -> bool:
        """Check the equality of `MultiIndexSet` instances.

        Parameters
        ----------
        other : `MultiIndexSet`
            The second operand of the equality check.

        Returns
        -------
        bool
            ``True`` if the two instances are equal in value, i.e., have
            the same underlying exponents and :math:`l_p`-degree value, and
            ``False`` otherwise.
        """
        return (
            self.lp_degree == other.lp_degree and
            np.array_equal(self.exponents, other.exponents)
        )

    def __mul__(self, other: "MultiIndexSet") -> "MultiIndexSet":
        """Multiply an instance of `MultiIndexSet` with another.

        Parameters
        ----------
        other : `MultiIndexSet`
            The second operand of the multi-index set multiplication.

        Returns
        -------
        `MultiIndexSet`
            The product of two multi-index sets.
            If the :math:`l_p`-degrees of the operands are different,
            then the larger :math:`l_p`-degree becomes the :math:`l_p`-degree
            of the product set.
            If the dimension differs, the product set has the dimension
            of the set with the larger dimension.
        """
        return self.multiply(other, inplace=False)

    def __imul__(self, other: "MultiIndexSet") -> "MultiIndexSet":
        """Multiply inplace an instance of `MultiIndexSet` with another.

        Parameters
        ----------

        Returns
        -------
        """
        # Multiply and modify the instance
        self.multiply(other, inplace=True)

        return self

    def __or__(self, other: "MultiIndexSet") -> "MultiIndexSet":
        """Combine an instance of `MultiIndexSet` with another via op.

        Parameters
        ----------
        other : `MultiIndexSet`
            The second operand of the multi-index set union.

        Returns
        -------
        `MultiIndexSet`
            The union of two multi-index sets.
            If the :math:`l_p`-degrees of the operands are different,
            then the larger :math:`l_p`-degree becomes the :math:`l_p`-degree
            of the union set.
            If the dimension differs, the union set has the dimension
            of the set with the larger dimension.
        """
        return self.union(other, inplace=False)

    def __ior__(self, other: "MultiIndexSet") -> "MultiIndexSet":
        """Combine an instance of `MultiIndexSet` with another inplace via op.

        Parameters
        ----------
        other : `MultiIndexSet`
            The second operand of the multi-index set union.

        Returns
        -------
        `MultiIndexSet`
            The union of two multi-index sets, updating the left-hand-side
            operand. If the :math:`l_p`-degrees of the operands are different,
            then the larger :math:`l_p`-degree becomes the :math:`l_p`-degree
            of the union set.
            If the dimension differs, the union set has the dimension
            of the set with the larger dimension.
        """
        self.union(other, inplace=True)

        return self

    # TODO make_derivable(): add (only) partial derivative exponent vectors,
    # NOTE: not meaningful since derivation requires complete index sets anyway?
