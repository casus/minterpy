"""
Abstract base class for the various polynomial classes.
"""
import abc
from copy import deepcopy
from typing import Union, Optional

import numpy as np

from minterpy.global_settings import ARRAY
from minterpy.grid import Grid
from minterpy.multi_index import MultiIndex

__all__ = ['MultivariatePolynomialABC', 'MultivariatePolynomialSingleABC']

from minterpy.multi_index_utils import find_match_between
from minterpy.verification import check_type_n_values, check_shape, verify_domain


class MultivariatePolynomialABC(abc.ABC):
    """ the most general abstract base class for multivariate polynomials

    defining the attributes and functions a multivariate polynomial must at least have
    """

    @property
    @abc.abstractmethod
    def coeffs(self) -> ARRAY:
        pass

    @coeffs.setter
    def coeffs(self, value):
        pass

    @property
    @abc.abstractmethod
    def nr_of_monomials(self):
        """
        NOTE: this is usually equal to the "amount of coefficients".
            However the coefficients can also be a 2D array
            (representing a multitude of polynomials with the same base grid)
        :return: the amount of monomials
        """
        pass

    @property
    @abc.abstractmethod
    def spatial_dimension(self):
        """ the dimensionality of the polynomial
        """
        pass

    @property
    @abc.abstractmethod
    def unisolvent_nodes(self):
        """ the points the polynomial is defined on
        """
        pass

    @abc.abstractmethod
    def _eval(self, arg) -> Union[float, ARRAY]:
        pass

    # TODO *args, **kwargs ?! or rather "point" or "x"
    def __call__(self, arg) -> Union[float, ARRAY]:
        """  NOTE: the output may be a ndarray when multiple sets of coefficients have been stored

        :param arg:
        :return:
        """
        # TODO built in rescaling between user_domain and internal_domain
        #   IDEA: use sklearn min max scaler (transform() and inverse_transform())
        return self._eval(arg)

    # anything else any polynomial must support
    # TODO mathematical operations? abstract
    # TODO copy operations. abstract


class MultivariatePolynomialSingleABC(MultivariatePolynomialABC):
    """ abstract base class for "single instance" multivariate polynomials

    the grid with the corresponding indices defines the "basis" or polynomial space a polynomial is part of.
    e.g. also the constraints for a Lagrange polynomial, i.e. on which points they must vanish.
    ATTENTION: the grid might be defined on other indices than multi_index!
      e.g. useful for defining Lagrange coefficients with "extra constraints"
      but all indices from multi_index must be contained in the grid!
      this corresponds to polynomials with just some of the Lagrange polynomials of the basis being "active"
    """
    _coeffs: Optional[ARRAY] = None

    @staticmethod
    @abc.abstractmethod
    def generate_internal_domain(internal_domain, spatial_dimension):
        pass

    @staticmethod
    @abc.abstractmethod
    def generate_user_domain(user_domain, spatial_dimension):
        pass

    # TODO static methods should not have a parameter "self"
    @staticmethod
    @abc.abstractmethod
    def _add(self, other):
        pass

    @staticmethod
    @abc.abstractmethod
    def _sub(self, other):
        pass

    @staticmethod
    @abc.abstractmethod
    def _mul(self, other):
        pass

    @staticmethod
    @abc.abstractmethod
    def _div(self, other):
        pass

    @staticmethod
    @abc.abstractmethod
    def _pow(self, pow):
        pass

    @staticmethod
    def _gen_grid_default(multi_index):
        return Grid(multi_index)

    def __init__(self, coeffs: Optional[ARRAY],
                 multi_index: Union[MultiIndex, ARRAY],
                 internal_domain: Optional[ARRAY] = None,
                 user_domain: Optional[ARRAY] = None,
                 grid: Optional[Grid] = None):

        if multi_index.__class__ is MultiIndex:
            self.multi_index = multi_index
        else:
            # TODO should passing multi indices as ndarray be supported?
            check_type_n_values(multi_index)  # expected ARRAY
            check_shape(multi_index, dimensionality=2)
            self.multi_index = MultiIndex(multi_index)

        nr_monomials, spatial_dimension = self.multi_index.exponents.shape
        self.coeffs = coeffs  # calls the setter method and checks the input shape

        if internal_domain is not None:
            check_type_n_values(internal_domain)
            check_shape(internal_domain, (spatial_dimension, 2))
        self.internal_domain = self.generate_internal_domain(internal_domain, self.multi_index.spatial_dimension)

        if user_domain is not None:  # TODO not better "external domain"?!
            check_type_n_values(user_domain)
            check_shape(user_domain, (spatial_dimension, 2))
        self.user_domain = self.generate_user_domain(user_domain, self.multi_index.spatial_dimension)

        # TODO make multi_index input optional? otherwise use the indices from grid
        # TODO class method from_grid
        if grid is None:
            grid = self._gen_grid_default(self.multi_index)
        if type(grid) is not Grid:
            raise ValueError(f'unexpected type {type(grid)} of the input grid')

        if not grid.multi_index.is_super_index_set_of(self.multi_index):
            raise ValueError("the multi indices of a polynomial must be a subset of the indices of the grid in use")
        self.grid: Grid = grid
        # weather or not the indices are independent from the grid ("basis")
        self.indices_are_separate: bool = self.grid.multi_index is not self.multi_index
        self.index_correspondence: Optional[ARRAY] = None  # 1:1 correspondence
        if self.indices_are_separate:
            # store the position of the active Lagrange polynomials with respect to the basis indices:
            self.index_correspondence = find_match_between(self.multi_index.exponents, self.grid.multi_index.exponents)

    @classmethod
    def from_degree(cls, coeffs: Optional[ARRAY], spatial_dimension: int, poly_degree: int, lp_degree: int,
                    internal_domain: ARRAY = None, user_domain: ARRAY = None):
        return cls(coeffs, MultiIndex.from_degree(spatial_dimension, poly_degree, lp_degree), internal_domain,
                   user_domain)

    @classmethod
    def from_poly(cls, polynomial: 'MultivariatePolynomialSingleABC',
                  new_coeffs: Optional[ARRAY] = None) -> 'MultivariatePolynomialSingleABC':
        """ constructs a new polynomial instance based on the properties of an input polynomial

        useful for copying polynomials of other types
        NOTE: the coefficients can also be assigned later

        :param polynomial: input polynomial instance defining the properties to be reused
        :param new_coeffs: the coefficients the new polynomials should have. using `polynomial.coeffs` if `None`
        :return: new polynomial instance with equal properties
        """
        p = polynomial
        if new_coeffs is None:  # use the same coefficients
            new_coeffs = p.coeffs
        return cls(new_coeffs, p.multi_index, p.internal_domain, p.user_domain, p.grid)

    # Arithmetic operations:

    def __neg__(self):
        return self.__class__(-self._coeffs, self.multi_index, self.internal_domain, self.user_domain)

    def __pos__(self):
        return self

    def __add__(self, other):
        if self.__class__ != other.__class__:
            raise NotImplementedError(f"Addition operation not implemented for "
                                      f"'{self.__class__}', '{other.__class__}'")

        result = self._add(self, other)
        return result

    def __sub__(self, other):
        if self.__class__ != other.__class__:
            raise NotImplementedError(f"Subtraction operation not implemented for "
                                      f"'{self.__class__}', '{other.__class__}'")

        result = self._sub(self, other)
        return result

    def __mul__(self, other):
        if self.__class__ != other.__class__:
            raise NotImplementedError(f"Multiplication operation not implemented for "
                                      f"'{self.__class__}', '{other.__class__}'")

        result = self._mul(self, other)
        return result

    def __radd__(self, other):
        if self.__class__ != other.__class__:
            raise NotImplementedError(f"Addition operation not implemented for "
                                      f"'{self.__class__}', '{other.__class__}'")

        result = self._add(other, self)
        return result

    def __rsub__(self, other):
        if self.__class__ != other.__class__:
            raise NotImplementedError(f"Subtraction operation not implemented for "
                                      f"'{self.__class__}', '{other.__class__}'")

        result = self._add(-other, self)
        return result

    def __rmul__(self, other):
        if self.__class__ != other.__class__:
            raise NotImplementedError(f"Multiplication operation not implemented for "
                                      f"'{self.__class__}', '{other.__class__}'")

        # TODO Call to the _mul method
        # TODO Return the a new class instance with the result
        return

    # copying
    def __copy__(self):
        return self.__class__(self._coeffs, self.multi_index, self.internal_domain, self.user_domain, self.grid)

    def __deepcopy__(self, mem):
        return self.__class__(deepcopy(self._coeffs), deepcopy(self.multi_index), deepcopy(self.internal_domain),
                              deepcopy(self.user_domain), deepcopy(self.grid))

    @property
    def nr_of_monomials(self):
        """
        NOTE: this is usually equal to the "amount of coefficients".
            However the coefficients can also be a 2D array
            (representing a multitude of polynomials with the same base grid)
        :return: the amount of monomials
        """
        return len(self.multi_index)

    @property
    def spatial_dimension(self):
        """ the dimensionality of the polynomial
        """
        return self.multi_index.spatial_dimension

    @property
    def coeffs(self) -> Optional[ARRAY]:
        """
        :returns (N) or (N, p) the coefficients of the multivariate polynomial(s).
            N = amount of monomials
            p = amount of polynomials
        """
        if self._coeffs is None:
            raise ValueError('trying to access an uninitialized polynomial (coefficients are `None`)')
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value: Optional[ARRAY]):
        """
        :param value: (N) or (N, p) the coefficients of the multivariate polynomial(s).
            N = amount of monomials
            p = amount of polynomials
            NOTE: it is allowed to set the coefficients to `None` to represent a not yet initialised polynomial
        """
        if value is None:
            self._coeffs = None
            return
        check_type_n_values(value)
        if value.shape[0] != self.nr_of_monomials:
            raise ValueError(
                f"the amount of given coefficients <{value.shape[0]}> does not match "
                f"with the amount of monomials in the polynomial <{self.nr_of_monomials}>.")
        self._coeffs = value

    @property
    def unisolvent_nodes(self):
        """ the points the polynomial is defined on
        """
        return self.grid.unisolvent_nodes

    def _new_instance_if_necessary(self, new_indices: MultiIndex) -> 'MultivariatePolynomialSingleABC':
        """ constructs new class only if the multi indices have changed
        """
        old_indices = self.multi_index
        if new_indices is old_indices:
            return self
        if not old_indices.is_sub_index_set_of(new_indices):
            raise ValueError('an index set of a polynomial can only be expanded, '
                             'but the old indices contain multi indices not present in the new indices.')

        # convert the coefficients correctly:
        if self._coeffs is None:
            new_coeffs = None
        else:
            new_coeffs = np.zeros(len(new_indices))
            idxs_of_old = find_match_between(old_indices.exponents, new_indices.exponents)
            new_coeffs[idxs_of_old] = self._coeffs
        # replace the grid with an independent copy with new multi indices
        # ATTENTION: the grid might be defined on other indices than multi_index!
        #   but all indices from multi_index must be contained in the grid!
        # -> make sure to add all new additional indices also to the grid!
        new_grid = self.grid.add_points(new_indices.exponents)
        new_poly_instance = self.__class__(new_coeffs, new_indices, grid=new_grid)
        return new_poly_instance

    def make_complete(self) -> "MultivariatePolynomialSingleABC":
        """ convert the polynomial into a new polynomial instance with a complete multi index set

        NOTE: (only) in the case of a Lagrange polynomial this could be done
            by evaluating the polynomial on the complete grid
        """
        # TODO only grid
        new_indices = self.multi_index.make_complete()
        # ATTENTION: also the grid ("basis") needs to be completed!
        return self._new_instance_if_necessary(new_indices)

    def add_points(self, exponents: ARRAY) -> 'MultivariatePolynomialSingleABC':
        multi_indices_new = self.multi_index.add_exponents(exponents)
        return self._new_instance_if_necessary(multi_indices_new)

    # def make_derivable(self) -> "MultivariatePolynomialSingleABC":
    #     """ convert the polynomial into a new polynomial instance with a "derivable" multi index set
    #  NOTE: not meaningful since derivation requires complete index sets anyway?
    #     """
    #     new_indices = self.multi_index.make_derivable()
    #     return self._new_instance_if_necessary(new_indices)

    def expand_dim(self, dim, extra_internal_domain=None, extra_user_domain=None):
        """
        Expands the dimension of the polynomial by adding zeros to the multi_indices
        (which is equivalent to the multiplication of ones to each monomial)

        TODO handle grid points.
        """
        expand_dim = dim - self.multi_index.spatial_dimension

        self.multi_index.expand_dim(dim)  # breaks if dim<spacial_dimension, i.e. expand_dim<0
        extra_internal_domain = verify_domain(extra_internal_domain, expand_dim)
        self.internal_domain = np.concatenate((self.internal_domain, extra_internal_domain))
        extra_user_domain = verify_domain(extra_user_domain, expand_dim)
        self.user_domain = np.concatenate((self.user_domain, extra_user_domain))