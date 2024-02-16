"""
Abstract base class for the various polynomial base classes.

This module contains the abstract base classes, from which all concrete implementations of polynomial classes shall subclass.
This ensures that all polynomials work with the same interface, so futher features can be formulated without referencing the concrete polynomial implementation. See e.g. :PEP:`3119` for further explanations on that topic.
"""
import abc
from copy import deepcopy
from typing import Any, List, Optional, Union

import numpy as np

from minterpy.global_settings import ARRAY

from ..grid import Grid
from ..multi_index import MultiIndexSet
from ..utils import expand_dim, find_match_between
from ..verification import (
    check_dimensionality,
    check_shape,
    check_type_n_values,
    verify_domain,
)

__all__ = ["MultivariatePolynomialABC", "MultivariatePolynomialSingleABC"]


class MultivariatePolynomialABC(abc.ABC):
    """the most general abstract base class for multivariate polynomials.

    Every data type which needs to behave like abstract polynomial(s) should subclass this class and implement all the abstract methods.
    """

    @property
    @abc.abstractmethod
    def coeffs(self) -> ARRAY:  # pragma: no cover
        """Abstract container which stores the coefficients of the polynomial.

        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @coeffs.setter
    def coeffs(self, value):
        pass

    @property
    @abc.abstractmethod
    def nr_active_monomials(self):  # pragma: no cover
        """Abstract container for the number of monomials of the polynomial(s).

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @property
    @abc.abstractmethod
    def spatial_dimension(self):  # pragma: no cover
        """Abstract container for the dimension of space where the polynomial(s) live on.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @property
    @abc.abstractmethod
    def unisolvent_nodes(self):  # pragma: no cover
        """Abstract container for unisolvent nodes the polynomial(s) is(are) defined on.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @abc.abstractmethod
    def _eval(self, x) -> Any:  # pragma: no cover
        """Abstract evaluation function.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    # TODO *args, **kwargs ?! or rather "point" or "x"
    def __call__(self, x) -> Any:
        """Evaluation of the polynomial.

        This function is called, if an instance of the polynomial(s) is called: ``P(x)``

        Parameters
        ----------
        arg : np.ndarray
            Batch array where the polynomial(s) is evaluated. The shape needs to be ``(N,d)`` or ``(N,d,n)``, where ``N`` is the number of points, ``d`` is the dimension of the space and ``n`` is the number of polynomials (if present). Actually, ``arg`` denotes list of `d`-tuples containing the components of the point in space.

        Returns
        -------
        np.ndarray
            The output array of the scalar values of the polynomials the output may be a ndarray when multiple sets of coefficients have been stored

        Notes
        -----
        Internally the concrete implementation of the static method ``_eval`` is called.

        See Also
        --------
        _eval : concrete implementation of the evaluation of the polynomial(s)
        """
        # TODO built in rescaling between user_domain and internal_domain
        #   IDEA: use sklearn min max scaler (transform() and inverse_transform())
        return self._eval(x)

    # anything else any polynomial must support
    # TODO mathematical operations? abstract
    # TODO copy operations. abstract


class MultivariatePolynomialSingleABC(MultivariatePolynomialABC):
    """abstract base class for "single instance" multivariate polynomials

    Attributes
    ----------
    multi_index : MultiIndexSet
        The multi-indices of the multivariate polynomial.
    internal_domain : array_like
        The domain the polynomial is defined on (basically the domain of the unisolvent nodes).
        Either one-dimensional domain (min,max), a stack of domains for each
        domain with shape (spatial_dimension,2).
    user_domain : array_like
        The domain where the polynomial can be evaluated. This will be mapped onto the ``internal_domain``.
        Either one-dimensional domain ``min,max)`` a stack of domains for each
        domain with shape ``(spatial_dimension,2)``.

    Notes
    -----
    the grid with the corresponding indices defines the "basis" or polynomial space a polynomial is part of.
    e.g. also the constraints for a Lagrange polynomial, i.e. on which points they must vanish.
    ATTENTION: the grid might be defined on other indices than multi_index! e.g. useful for defining Lagrange coefficients with "extra constraints"
    but all indices from multi_index must be contained in the grid!
    this corresponds to polynomials with just some of the Lagrange polynomials of the basis being "active"
    """

    # __doc__ += __doc_attrs__

    _coeffs: Optional[ARRAY] = None

    @staticmethod
    @abc.abstractmethod
    def generate_internal_domain(
        internal_domain, spatial_dimension
    ):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def generate_user_domain(user_domain, spatial_dimension):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    # TODO static methods should not have a parameter "self"
    @staticmethod
    @abc.abstractmethod
    def _add(self, other):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _sub(self, other):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _mul(self, other):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _div(self, other):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _pow(self, pow):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    def _gen_grid_default(multi_index):
        """Return the default :class:`Grid` for a given :class:`MultiIndexSet` instance.

        For the default values of the Grid class, see :class:`minterpy.Grid`.


        :param multi_index: An instance of :class:`MultiIndexSet` for which the default :class:`Grid` shall be build
        :type multi_index: MultiIndexSet
        :return: An instance of :class:`Grid` with the default optional parameters.
        :rtype: Grid
        """
        return Grid(multi_index)

    @staticmethod
    @abc.abstractmethod
    def _integrate_over(
        poly: "MultivariatePolynomialABC", bounds: Optional[np.ndarray]
    ) -> np.ndarray:
        """Abstract definite integration method."""
        pass

    def __init__(
        self,
        multi_index: Union[MultiIndexSet, ARRAY],
        coeffs: Optional[ARRAY] = None,
        internal_domain: Optional[ARRAY] = None,
        user_domain: Optional[ARRAY] = None,
        grid: Optional[Grid] = None,
    ):

        if multi_index.__class__ is MultiIndexSet:
            if len(multi_index) == 0:
                raise ValueError("MultiIndexSet must not be empty!")
            self.multi_index = multi_index
        else:
            # TODO should passing multi indices as ndarray be supported?
            check_type_n_values(multi_index)  # expected ARRAY
            check_dimensionality(multi_index, dimensionality=2)
            self.multi_index = MultiIndexSet(multi_index)

        nr_monomials, spatial_dimension = self.multi_index.exponents.shape
        self.coeffs = coeffs  # calls the setter method and checks the input shape

        if internal_domain is not None:
            check_type_n_values(internal_domain)
            check_shape(internal_domain, shape=(spatial_dimension, 2))
        self.internal_domain = self.generate_internal_domain(
            internal_domain, self.multi_index.spatial_dimension
        )

        if user_domain is not None:  # TODO not better "external domain"?!
            check_type_n_values(user_domain)
            check_shape(user_domain, shape=(spatial_dimension, 2))
        self.user_domain = self.generate_user_domain(
            user_domain, self.multi_index.spatial_dimension
        )

        # TODO make multi_index input optional? otherwise use the indices from grid
        # TODO class method from_grid
        if grid is None:
            grid = self._gen_grid_default(self.multi_index)
        if type(grid) is not Grid:
            raise ValueError(f"unexpected type {type(grid)} of the input grid")

        if not grid.multi_index.is_superset(self.multi_index):
            raise ValueError(
                "the multi indices of a polynomial must be a subset of the indices of the grid in use"
            )
        self.grid: Grid = grid
        # weather or not the indices are independent from the grid ("basis")
        # TODO this could be enconded by .active_monomials being None
        self.indices_are_separate: bool = self.grid.multi_index is not self.multi_index
        self.active_monomials: Optional[ARRAY] = None  # 1:1 correspondence
        if self.indices_are_separate:
            # store the position of the active Lagrange polynomials with respect to the basis indices:
            self.active_monomials = find_match_between(
                self.multi_index.exponents, self.grid.multi_index.exponents
            )

    @classmethod
    def from_degree(
        cls,
        spatial_dimension: int,
        poly_degree: int,
        lp_degree: int,
        coeffs: Optional[ARRAY] = None,
        internal_domain: ARRAY = None,
        user_domain: ARRAY = None,
    ):
        """Initialise Polynomial from given coefficients and the default construction for given polynomial degree, spatial dimension and :math:`l_p` degree.

        :param spatial_dimension: Dimension of the domain space of the polynomial.
        :type spatial_dimension: int

        :param poly_degree: The degree of the polynomial, i.e. the (integer) supremum of the :math:`l_p` norms of the monomials.
        :type poly_degree: int

        :param lp_degree: The :math:`l_p` degree used to determine the polynomial degree.
        :type lp_degree: int

        :param coeffs: coefficients of the polynomial. These shall be 1D for a single polynomial, where the length of the array is the number of monomials given by the ``multi_index``. For a set of similar polynomials (with the same number of monomials) the array can also be 2D, where the first axis refers to the monomials and the second axis refers to the polynomials.
        :type coeffs: np.ndarray

        :param internal_domain: the internal domain (factory) where the polynomials are defined on, e.g. :math:`[-1,1]^d` where :math:`d` is the dimension of the domain space. If a ``callable`` is passed, it shall get the dimension of the domain space and returns the ``internal_domain`` as an :class:`np.ndarray`.
        :type internal_domain: np.ndarray or callable
        :param user_domain: the domain window (factory), from which the arguments of a polynomial are transformed to the internal domain. If a ``callable`` is passed, it shall get the dimension of the domain space and returns the ``user_domain`` as an :class:`np.ndarray`.
        :type user_domain: np.ndarray or callable

        """
        return cls(
            MultiIndexSet.from_degree(spatial_dimension, poly_degree, lp_degree),
            coeffs,
            internal_domain,
            user_domain,
        )

    @classmethod
    def from_poly(
        cls,
        polynomial: "MultivariatePolynomialSingleABC",
        new_coeffs: Optional[ARRAY] = None,
    ) -> "MultivariatePolynomialSingleABC":
        """constructs a new polynomial instance based on the properties of an input polynomial

        useful for copying polynomials of other types


        :param polynomial: input polynomial instance defining the properties to be reused
        :param new_coeffs: the coefficients the new polynomials should have. using `polynomial.coeffs` if `None`
        :return: new polynomial instance with equal properties

        Notes
        -----
        The coefficients can also be assigned later.
        """
        p = polynomial
        if new_coeffs is None:  # use the same coefficients
            new_coeffs = p.coeffs

        return cls(p.multi_index, new_coeffs, p.internal_domain, p.user_domain, p.grid)

    # Arithmetic operations:

    def __neg__(self):
        """
        Negation of the polynomial(s).

        :return: new polynomial with negated coefficients.
        :rtype: MultivariatePolynomialSingleABC
        """
        return self.__class__(
            self.multi_index, -self._coeffs, self.internal_domain, self.user_domain
        )

    def __pos__(self):
        """
        Plus signing of polynomial(s).

        :return: the same polynomial
        :rtype: MultivariatePolynomialSingleABC
        """
        return self

    def __add__(self, other):
        """Addition of the polynomial(s) with another given polynomial(s).

        This function is called, if two polynomials are added like ``P1 + P2``.

        :param other: the other polynomial, which is added.
        :type other: MultivariatePolynomialSingleABC

        :return: The result of ``self + other``.
        :rtype: MultivariatePolynomialSingleABC

        Notes
        -----
        Internally it calles the static method ``self._add`` from the concrete implementation.

        See Also
        --------
        _add : concrete implementation of ``__add__``
        """
        if self.__class__ != other.__class__:
            raise NotImplementedError(
                f"Addition operation not implemented for "
                f"'{self.__class__}', '{other.__class__}'"
            )

        result = self._add(self, other)
        return result

    def __sub__(self, other):
        """Substraction of the polynomial(s) with another given polynomial(s).

        This function is called, if two polynomials are substracted like ``P1 - P2``.

        :param other: the other polynomial, which is substracted.
        :type other:  MultivariatePolynomialSingleABC

        :return: The result of ``self - other``.
        :rtype: MultivariatePolynomialSingleABC

        Notes
        -----
        Internally it calles the static method ``self._sub`` from the concrete implementation.

        See Also
        --------
        _sub : concrete implementation of ``__sub__``
        """
        if self.__class__ != other.__class__:
            raise NotImplementedError(
                f"Subtraction operation not implemented for "
                f"'{self.__class__}', '{other.__class__}'"
            )

        result = self._sub(self, other)
        return result

    def __mul__(self, other):
        """Multiplication of the polynomial(s) with another given polynomial(s).

        This function is called, if two polynomials are multiplied like ``P1 * P2``.

        :param other: the other polynomial, which is multiplied.
        :type other:  MultivariatePolynomialSingleABC

        :return: The result of ``self * other``.
        :rtype: MultivariatePolynomialSingleABC

        Notes
        -----
        Internally it calles the static method ``self._mul`` from the concrete implementation.

        See Also
        --------
        _mul : concrete implementation of ``__mul__``
        """
        if self.__class__ != other.__class__:
            raise NotImplementedError(
                f"Multiplication operation not implemented for "
                f"'{self.__class__}', '{other.__class__}'"
            )

        result = self._mul(self, other)
        return result

    def __radd__(self, other):
        """Right sided addition of the polynomial(s) with another given polynomial(s).

        This function is called, if two polynomials are added like ``P1 + P2`` from the right side.

        :param other: The other polynomial, where this instance is added on.
        :type other:  MultivariatePolynomialSingleABC

        :return: The result of ``other + self``.
        :rtype: MultivariatePolynomialSingleABC

        Notes
        -----
        Internally it calles the static method ``self._radd`` from the concrete implementation.

        See Also
        --------
        _radd : concrete implementation of ``__radd__``
        """
        if self.__class__ != other.__class__:
            raise NotImplementedError(
                f"Addition operation not implemented for "
                f"'{self.__class__}', '{other.__class__}'"
            )

        result = self._add(other, self)
        return result

    def __rsub__(self, other):
        """Right sided difference of the polynomial(s) with another given polynomial(s).

        This function is called, if two polynomials are substracted like ``P1 - P2`` from the right side.

        :param other: The other polynomial, where this instance is substracted from.
        :type other: MultivariatePolynomialSingleABC

        :return: The result of ``other - self``.
        :rtype: MultivariatePolynomialSingleABC

        Notes
        -----
        Internally it calles the static method ``self._rsub`` from the concrete implementation.

        See Also
        --------
        _rsub : concrete implementation of ``__rsub__``
        """
        if self.__class__ != other.__class__:
            raise NotImplementedError(
                f"Subtraction operation not implemented for "
                f"'{self.__class__}', '{other.__class__}'"
            )

        result = self._add(-other, self)
        return result

    def __rmul__(self, other):
        """Right sided multiplication of the polynomial(s) with another given polynomial(s).

        This function is called, if two polynomials are multiplied like ``P1*P2`` from the right side.

        :param other: The other polynomial, where this instance is multplied on.
        :type other: MultivariatePolynomialSingleABC

        :return: The result of ``other * self``.
        :rtype: MultivariatePolynomialSingleABC

        Notes
        -----
        Internally it calles the static method ``self._rmul`` from the concrete implementation.

        See Also
        --------
        _rmul : concrete implementation of ``__rmul__``
        """
        if self.__class__ != other.__class__:
            raise NotImplementedError(
                f"Multiplication operation not implemented for "
                f"'{self.__class__}', '{other.__class__}'"
            )

        # TODO Call to the _mul method
        # TODO Return the a new class instance with the result
        return

    # copying
    def __copy__(self):
        """Creates of a shallow copy.

        This function is called, if one uses the top-level function ``copy()`` on an instance of this class.

        :return: The copy of the current instance.
        :rtype: MultivariatePolynomialSingleABC

        See Also
        --------
        copy.copy
            copy operator form the python standard library.
        """
        return self.__class__(
            self.multi_index,
            self._coeffs,
            self.internal_domain,
            self.user_domain,
            self.grid,
        )

    def __deepcopy__(self, mem):
        """Creates of a deepcopy.

        This function is called, if one uses the top-level function ``deepcopy()`` on an instance of this class.

        :return: The deepcopy of the current instance.
        :rtype: MultivariatePolynomialSingleABC

        See Also
        --------
        copy.deepcopy
            copy operator form the python standard library.

        """
        return self.__class__(
            deepcopy(self.multi_index),
            deepcopy(self._coeffs),
            deepcopy(self.internal_domain),
            deepcopy(self.user_domain),
            deepcopy(self.grid),
        )

    @property
    def nr_active_monomials(self):
        """Number of active monomials of the polynomial(s).

        For caching and methods based on switching single monomials on and off, it is distigushed between active and passive monomials, where only the active monomials particitpate on exposed functions.

        :return: Number of active monomials.
        :rtype: int

        Notes
        -----
        This is usually equal to the "amount of coefficients". However the coefficients can also be a 2D array (representing a multitude of polynomials with the same base grid).
        """

        return len(self.multi_index)

    @property
    def spatial_dimension(self):
        """Spatial dimension.

        The dimension of space where the polynomial(s) live on.

        :return: Dimension of domain space.
        :rtype: int

        Notes
        -----
        This is propagated from the ``multi_index.spatial_dimension``.
        """
        return self.multi_index.spatial_dimension

    @property
    def coeffs(self) -> Optional[ARRAY]:
        """Array which stores the coefficients of the polynomial.

        With shape (N,) or (N, p) the coefficients of the multivariate polynomial(s), where N is the amount of monomials and p is the amount of polynomials.

        :return: Array of coefficients.
        :rtype: np.ndarray
        :raise ValueError: Raised if the coeffs are not initialised.

        Notes
        -----
        It is allowed to set the coefficients to `None` to represent a not yet initialised polynomial
        """
        if self._coeffs is None:
            raise ValueError(
                "trying to access an uninitialized polynomial (coefficients are `None`)"
            )
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value: Optional[ARRAY]):
        # setters shall not have docstrings. See numpydoc class example.
        if value is None:
            self._coeffs = None
            return
        check_type_n_values(value)
        if value.shape[0] != self.nr_active_monomials:
            raise ValueError(
                f"the amount of given coefficients <{value.shape[0]}> does not match "
                f"with the amount of monomials in the polynomial <{self.nr_active_monomials}>."
            )
        self._coeffs = value

    @property
    def unisolvent_nodes(self):
        """Unisolvent nodes the polynomial(s) is(are) defined on.

        For definitions of unisolvent nodes see the mathematical introduction.

        :return: Array of unisolvent nodes.
        :rtype: np.ndarray

        Notes
        -----
        This is propagated from from ``self.grid.unisolvent_nodes``.
        """
        return self.grid.unisolvent_nodes

    def _new_instance_if_necessary(
        self, new_grid, new_indices: Optional[MultiIndexSet] = None
    ) -> "MultivariatePolynomialSingleABC":
        """Constructs a new instance only if the multi indices have changed.

        :param new_grid: Grid instance the polynomial is defined on.
        :type new_grid: Grid

        :param new_indices: :class:`MultiIndexSet` instance for the polynomial(s), needs to be a subset of the current ``multi_index``. Default is :class:`None`.
        :type new_indices: MultiIndexSet, optional

        :return: Same polynomial instance if ``grid`` and ``multi_index`` stay the same, otherwise new polynomial instance with the new ``grid`` and ``multi_index``.
        :rtype: MultivariatePolynomialSingleABC
        """
        prev_grid = self.grid
        if new_grid is prev_grid:
            return self
        # grid has changed
        if new_indices is None:
            # the active monomials (and coefficients) stay equal
            new_indices = self.multi_index
            new_coeffs = self._coeffs
        else:
            # also the active monomials change
            prev_indices = self.multi_index
            if not prev_indices.is_subset(new_indices):
                raise ValueError(
                    "an index set of a polynomial can only be expanded, "
                    "but the old indices contain multi indices not present in the new indices."
                )

            # convert the coefficients correctly:
            if self._coeffs is None:
                new_coeffs = None
            else:
                new_coeffs = np.zeros(len(new_indices))
                idxs_of_old = find_match_between(
                    prev_indices.exponents, new_indices.exponents
                )
                new_coeffs[idxs_of_old] = self._coeffs

        new_poly_instance = self.__class__(new_indices, new_coeffs, grid=new_grid)
        return new_poly_instance

    def make_complete(self) -> "MultivariatePolynomialSingleABC":
        """returns a possibly new polynomial instance with a complete multi index set.

        :return: completed polynomial, where additional coefficients setted to zero.
        :rtype: MultivariatePolynomialSingleABC

        Notes
        -----
        - the active monomials stay equal. only the grid ("basis") changes
        - in the case of a Lagrange polynomial this could be done by evaluating the polynomial on the complete grid
        """
        grid_completed = self.grid.make_complete()
        return self._new_instance_if_necessary(grid_completed)

    def add_points(self, exponents: ARRAY) -> "MultivariatePolynomialSingleABC":
        """Extend ``grid`` and ``multi_index``

        Adds points ``grid`` and exponents to ``multi_index`` related to a given set of additional exponents.

        :param exponents: Array of exponents added.
        :type exponents: np.ndarray

        :return: New polynomial with the added exponents.
        :rtype: MultivariatePolynomialSingleABC

        """
        # replace the grid with an independent copy with the new multi indices
        # ATTENTION: the grid might be defined on other indices than multi_index!
        #   but all indices from multi_index must be contained in the grid!
        # -> make sure to add all new additional indices also to the grid!
        grid_new = self.grid.add_points(exponents)
        multi_indices_new = None
        if self.indices_are_separate:
            multi_indices_new = self.multi_index.add_exponents(exponents)
        return self._new_instance_if_necessary(grid_new, multi_indices_new)

    # def make_derivable(self) -> "MultivariatePolynomialSingleABC":
    #     """ convert the polynomial into a new polynomial instance with a "derivable" multi index set
    #  NOTE: not meaningful since derivation requires complete index sets anyway?
    #     """
    #     new_indices = self.multi_index.make_derivable()
    #     return self._new_instance_if_necessary(new_indices)

    def expand_dim(
        self,
        dim: int,
        extra_internal_domain: ARRAY = None,
        extra_user_domain: ARRAY = None,
    ):
        """Expand the spatial dimention.

        Expands the dimension of the domain space of the polynomial by adding zeros to the multi_indices
        (which is equivalent to the multiplication of ones to each monomial).
        Furthermore, the grid is now embedded in the higher dimensional space by pinning the grid arrays to the origin in the additional spatial dimension.

        :param dim: Number of additional dimensions.
        :type dim: int
        """
        diff_dim = dim - self.multi_index.spatial_dimension

        # If dim<spatial_dimension, i.e. expand_dim<0, exception is raised
        self.multi_index = self.multi_index.expand_dim(dim)

        grid = self.grid
        new_gen_pts = expand_dim(grid.generating_points, dim)
        new_gen_vals = expand_dim(grid.generating_values.reshape(-1, 1), dim)

        self.grid = Grid(self.multi_index, new_gen_pts, new_gen_vals)

        extra_internal_domain = verify_domain(extra_internal_domain, diff_dim)
        self.internal_domain = np.concatenate(
            (self.internal_domain, extra_internal_domain)
        )
        extra_user_domain = verify_domain(extra_user_domain, diff_dim)
        self.user_domain = np.concatenate((self.user_domain, extra_user_domain))

    def partial_diff(self, dim: int, order: int = 1) -> "MultivariatePolynomialSingleABC":
        """Compute the polynomial that is the partial derivative along a dimension of specified order.

        Parameters
        ----------
        dim: spatial dimension along which to take the derivative
        order: order of partial derivative

        Returns
        -------
        a new polynomial instance that represents the partial derivative
        """

        # Guard rails for dim
        if not np.issubdtype(type(dim), np.integer):
            raise TypeError(f"dim <{dim}> must be an integer")

        if dim < 0 or dim >= self.spatial_dimension:
            raise ValueError(f"dim <{dim}> for spatial dimension <{self.spatial_dimension}>"
                             f" should be between 0 and {self.spatial_dimension-1}")

        # Guard rails for order
        if not np.issubdtype(type(dim), np.integer):
            raise TypeError(f"order <{order}> must be a non-negative integer")

        if order < 0:
            raise ValueError(f"order <{order}> must be a non-negative integer")

        return self._partial_diff(self, dim, order)

    def diff(self, order: np.ndarray) -> "MultivariatePolynomialSingleABC":
        """Compute the polynomial that is the partial derivative of a particular order along each dimension.

        Parameters
        ----------
        order: integer array specifying the order of derivative along each dimension

        Returns
        -------
        a new polynomial instance that represents the partial derivative
        """

        # convert 'order' to numpy 1d array if it isn't already. This allows type checking below.
        order = np.ravel(order)

        # Guard rails for order
        if not np.issubdtype(order.dtype.type, np.integer):
            raise TypeError(f"order of derivative <{order}> can only be non-negative integers")

        if np.any(order < 0):
            raise ValueError(f"order of derivative <{order}> cannot have negative values")

        if len(order) != self.spatial_dimension:
            raise ValueError(f"inconsistent number of elements in 'order' <{len(order)}>,"
                             f"expected <{self.spatial_dimension}> corresponding to each spatial dimension")

        return self._diff(self, order)

    def integrate_over(
        self, bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Compute the definite integral of the polynomial over the bounds.

        Parameters
        ----------
        bounds : Union[List[List[float]], np.ndarray], optional
            The bounds of the integral, an ``(M, 2)`` array where ``M``
            is the number of spatial dimensions. Each row corresponds to
            the bounds in a given dimension.
            If not given, then the canonical bounds [-1, 1]^M will be used
            instead.

        Returns
        -------
        Union[:py:class:`float`, :class:`numpy:numpy.ndarray`]
            The integral value of the polynomial over the given bounds.
            If only one polynomial is available, the return value is of
            a :py:class:`float` type.

        Raises
        ------
        ValueError
            If the bounds either of inconsistent shape or not in the [-1, 1]^M
            domain.

        TODO
        ----
        - The default fixed domain [-1, 1]^M may in the future be relaxed.
          In that case, the domain check below along with the concrete
          implementations for the poly. classes must be updated.
        """
        num_dim = self.spatial_dimension
        if bounds is None:
            # The canonical bounds are [-1, 1]^M
            bounds = np.ones((num_dim, 2))
            bounds[:, 0] *= -1

        if isinstance(bounds, list):
            bounds = np.atleast_2d(bounds)

        # --- Bounds verification
        # Shape
        if bounds.shape != (num_dim, 2):
            raise ValueError(
                "The bounds shape is inconsistent! "
                f"Given {bounds.shape}, expected {(num_dim, 2)}."
            )
        # Domain fit, i.e., in [-1, 1]^M
        if np.any(bounds < -1) or np.any(bounds > 1):
            raise ValueError("Bounds are outside [-1, 1]^M domain!")

        # --- Compute the integrals
        # If the lower and upper bounds are equal, immediately return 0
        if np.any(np.isclose(bounds[:, 0], bounds[:, 1])):
            return 0.0

        value = self._integrate_over(self, bounds)

        try:
            # One-element array (one set of coefficients), just return the item
            return value.item()
        except ValueError:
            return value
