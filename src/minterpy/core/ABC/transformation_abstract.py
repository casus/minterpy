"""
Abstract base class for defining transformations from one polynomial basis to another.

This module provides the abstract base class for polynomial basis transformations (origin to target) from
which all concrete implentations are derived.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..grid import Grid
from ..multi_index import MultiIndexSet
from .multivariate_polynomial_abstract import MultivariatePolynomialSingleABC
from .operator_abstract import OperatorABC

__all__ = ["TransformationABC"]


class TransformationABC(ABC):
    """this is the abstract base class for polynomial basis transformations.

    All polynomial basis transformers should be derived from this class to maintain a uniform interface for
     basis transformations.
    """

    available_transforms: Dict[Any, Any] = {}

    def __init__(self, origin_poly: MultivariatePolynomialSingleABC):
        if not isinstance(origin_poly, MultivariatePolynomialSingleABC):
            raise TypeError(f"<{origin_poly}> is not a Polynomial type.")
        if not isinstance(origin_poly, self.origin_type):
            raise TypeError(
                f"<{origin_poly}> is not of the expected type {self.origin_type}."
            )
        self.origin_poly = origin_poly

        # TODO check for index completeness
        # is complete: -> store the transformation fct once!?
        # TODO transformation fct as an attribute -> assign here (e.g. barycentric if complete)

        # TODO automatic make complete?
        # self.origin_poly = self.origin_poly.make_complete()
        # raise ValueError('some transformations only work for complete multi index sets!')

        self._transformation_operator: Optional[np.ndarray] = None

    @property
    def multi_index(self) -> MultiIndexSet:
        """The multi index set of the origin polynomial (and also the target polynomial)."""
        return self.origin_poly.multi_index

    @property
    def grid(self) -> Grid:
        """The grid of the origin polynomial."""
        return self.origin_poly.grid

    # TODO register the transformation classes to the available_transforms dictionary
    # TODO integrate function to retrieve the proper transformation (cf. transformation_utils.py)
    def __init_subclass__(cls, **kwargs):
        """Add a concrete implementation to the registry of available transformations"""
        super().__init_subclass__(**kwargs)
        cls.available_transforms[(cls.origin_type, cls.target_type)] = cls

    # TODO: remove argument. store origin poly once and reuse.
    #  otherwise the user could input an incompatible polynomial
    #  avoids ugly duplicate calls:
    #   l2n_transformation = TransformationLagrangeToNewton(lagrange_poly)
    #   newton_poly = l2n_transformation(lagrange_poly)
    # on the other hand avoid constructing multiple transformation objects for the same basis
    # just for transforming multiple different polynomioals! -> rather check the validity of the input polynomial basis
    def __call__(self, origin_poly: Optional[MultivariatePolynomialSingleABC] = None):
        """Transformation of polynomial.

        This function is called when an instance of transformation is called ``T()`` or ``T(x)``. If called without
        any arguments, the transformation is applied on the origin_poly used to construct the transformation. If an
        argument is passed, the transformation is applied on the instance passed.

        :param origin_poly: (optional)  an instance of the polynomial to be transformed.
        :return: an instance of the polynomial in the target basis.
        """
        if origin_poly is None:
            origin_poly = self.origin_poly
        # TODO check the validity of the basis (basis of the input poly during init, grids match)
        # TODO helper fcts equality of bases ("grid") __eq__
        elif type(origin_poly) != self.origin_type:
            raise TypeError(
                f"Input polynomial type <{type(origin_poly)}> differs from expected polynomial type <{self.origin_type}>"
            )
        # TODO unsafe. user could input wrong polynomial (e.g. different multi_index!)
        #   or even worse: with same multi index but different grid! (undetected!)
        #   -> test input match!
        return self._apply_transformation(origin_poly)

    @property
    @abstractmethod
    def origin_type(self):
        """Abstract container that stores the data type of the origin polynomial.

        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @property
    @abstractmethod
    def target_type(self):
        """Abstract container that stores the data type of the target polynomial.

        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @abstractmethod
    def _get_transformation_operator(self):
        """Abstract container for storing the transformation operator.

        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @property
    def transformation_operator(self) -> OperatorABC:
        """The polynomial basis transformation operator.

        :return: instance of the transformation operator

        Notes
        -----
        The transformation operator once constructed can be reused for transforming other polynomial instance of the
        origin_poly type, which have the same basis (and grid) as the origin_poly, to the target_poly type.
        """
        if self._transformation_operator is None:
            self._transformation_operator = self._get_transformation_operator()
        return self._transformation_operator

    @property
    def _target_indices(self) -> MultiIndexSet:
        """The MultiIndexSet of the target_poly.

        :return: the indices the target polynomial will have
        """

        # NOTE: poly.multi_index and poly.grid.multi_index might not be equal!
        # this is required since e.g. transforming a polynomial in Lagrange basis
        #    into Newton basis possibly "activates" all indices of the basis (grid).

        # TODO more specifically: only all "previous" = lexicographically smaller indices will be active
        #    -> same as completing only the active multi indices.
        # ATTENTION: the multi indices of the basis in use must stay equal!
        return self.origin_poly.grid.multi_index

    def _apply_transformation(self, origin_poly):
        """Transforms the polynomial to the target type of the transformation.

        :param origin_poly: instance of the polynomial to be transformed
        :return: an instance of the transformed polynomial in the target basis
        """
        # TODO discuss: is it meaningful to create a new polynomial instance every time?
        #  perhaps allow an optional output polynomial as parameter and then just update the coefficients?
        # NOTE: construct a new polynomial from the input polynomial in order to copy all relevant internal attributes!
        output_poly = self.target_type.from_poly(origin_poly)
        # ATTENTION: assign the correct expected multi indices!
        output_poly.multi_index = self._target_indices
        # NOTE: only then the coefficients can be assigned, since the shapes need to match with the indices!
        # NOTE: this is calling self.transformation_operator.__matmul__(origin_poly.coeffs)
        target_coeffs = self.transformation_operator @ origin_poly.coeffs
        output_poly.coeffs = target_coeffs
        return output_poly
