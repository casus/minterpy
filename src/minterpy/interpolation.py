"""
The `interpolation` module is part of `minterpy`.

It contains:
    - `interpolate`, the main interpolation function.
    - `Interpolant`, data type which acts like the function which interpolates a given function.
    - `Interpolator`, data type which builds all relevant parts for an interpolation and caches them.
"""
from typing import Callable, Optional

import attr

from .core import Grid, MultiIndexSet
from .dds import dds
from .polynomials import NewtonPolynomial
from .global_settings import DEFAULT_LP_DEG

__all__ = ["Interpolator", "Interpolant", "interpolate"]


class InterpolationError(Exception):
    """Exception raised if the interpolation went wrong."""

    pass


@attr.s(frozen=True, order=False, eq=False)
class Interpolator:
    """The construction class for interpolation.

    Data type which contains all relevant parts for interpolation and caches them.

    Attributes
    ----------
    spatial_dimension : dimension of the domain space.
    poly_degree : degree of the interpolation polynomials
    lp_degree : degree of the :math:`l_p` norm used to determine the `poly_degree`.
    multi_index : lexicographically complete multi index set build from `(spatial_dimension, poly_degree, lp_degree)`.
    grid : `Grid` instance build from `multi_index`.

    """

    spatial_dimension: int = attr.ib()
    poly_degree: int = attr.ib()
    lp_degree: int = attr.ib()

    multi_index = attr.ib(init=False, repr=False)
    grid = attr.ib(init=False, repr=False)

    @multi_index.default
    def __multi_index_default(self) -> MultiIndexSet:
        return MultiIndexSet.from_degree(
            self.spatial_dimension, self.poly_degree, self.lp_degree
        )

    @grid.default
    def __grid_default(self) -> Grid:
        return Grid(self.multi_index)

    def __call__(self, fct: Callable) -> Optional[NewtonPolynomial]:
        """Interpolate a given function.

        Builds a `NewtonPolynomial` which interpolates the given `fct`, where the precomuted setting of the current instance is used.

        :param fct: Function to be interpolated. Needs to be (numpy) universal function which shall be interpolated. If `arr` is an :class:`np.ndarray` with shape ``arr.shape == (N,spatial_dimension)``, the signature needs to be ``fct(arr) -> res``, where ``res`` is an :class:`np.ndarray` with shape ``(N,)``.
        :type fct: Callable

        :return: Interpolation polynomial in Newton form, which interpolates the function ``fct``, where the used divided difference scheme is build from ``self.multi_index`` and ``self.grid``.
        :rtype: NewtonPolynomial

        :raises InterpolationError: Raised if anything goes wrong with the interpolation.
        """
        try:
            fct_values = fct(self.grid.unisolvent_nodes)
            # NOTE: Don't use np.squeeze as DDS results may be of shape (1,1)
            interpol_coeffs = dds(fct_values, self.grid.tree).reshape(-1)
        except Exception as e:
            raise InterpolationError(e) from e

        return NewtonPolynomial(self.multi_index, interpol_coeffs)


@attr.s(frozen=True, order=False, eq=False)
class Interpolant:
    """Data type representing the result of an interpolation.

    Instances of this class can be used as functions, which interpolate a given function. Users who do not want to learn anything about neither polynomial interpolation nor bases in multivariate polynomial bases may use the instances of this class just as an interpolative representant of their function, which they can evaluate. (Other properties are conceivable too)

    Attributes
    ----------
    fct : Function to be interpolated. Needs to be (numpy) universal function which shall be interpolated. If `arr` is an :class:`np.ndarray` with shape ``arr.shape == (N,spatial_dimension)``, the signature needs to be ``fct(arr) -> res``, where ``res`` is an :class:`np.ndarray` with shape ``(N,)``.
    interpolator : Instance of :class:`Interpolator`, which represents the interpolation scheme to be used.
    """

    fct: Callable = attr.ib(repr=False)
    interpolator: Interpolator = attr.ib(repr=False)
    __interpolation_poly: NewtonPolynomial = attr.ib(init=False, repr=False)

    @__interpolation_poly.default
    def __interpolation_poly_default(self):
        return self.interpolator(self.fct)

    @classmethod
    def from_degree(cls, fct, spatial_dimension, poly_degree, lp_degree):
        """Custom constructor of an interpolant using dimensionality and degree parameter.

        :param fct: Function to be interpolated. Needs to be (numpy) universal function which shall be interpolated. If `arr` is an :class:`np.ndarray` with shape ``arr.shape == (N,spatial_dimension)``, the signature needs to be ``fct(arr) -> res``, where ``res`` is an :class:`np.ndarray` with shape ``(N,)``.
        :type fct: Callable

        :param spatial_dimension: dimension of the domain space.
        :type spatial_dimension: int
        :param poly_degree: degree of the interpolation polynomials
        :type poly_degree: int
        :param lp_degree: degree of the :math:`l_p` norm used to determine the `poly_degree`.
        :type lp_degree: int

        :return: The interpolant of ``fct`` using the default interpolator build from ``(spatial_dimension, poly_degree, lp_degree)``.
        :rtype: Interpolant
        """
        return cls(fct, Interpolator(spatial_dimension, poly_degree, lp_degree))

    @property
    def spatial_dimension(self):
        """Dimension of the domain space the interpolation polynomial lives on.

        This is the propagated attribute from ``self.interpolator``.

        :rtype: int
        """
        return self.interpolator.spatial_dimension

    @property
    def poly_degree(self):
        """Degree of the interpolation polynomial.

        This is the propagated attribute from ``self.interpolator``.

        :rtype: int
        """
        return self.interpolator.poly_degree

    @property
    def lp_degree(self):
        """Degree of the :math:`l_p` norm.

        This is the propagated attribute from ``self.interpolator``.

        :rtype: int
        """
        return self.interpolator.lp_degree

    def __call__(self, pts):
        """Evaulate the interpolant on a given array of points.

        :param pts: Array of points, where the shape needs to be ``pts.shape == (N,spatial_dimension)``,

        """
        return self.__interpolation_poly(pts)


def interpolate(fct, spatial_dimension, poly_degree, lp_degree=DEFAULT_LP_DEG):
    """Interpolate a given function.

    Return an interpolant, which represents the given function on the domain :math:`[-1, 1]^d`, where :math:`d` is the dimension of the domain space.



    :param fct: Function to be interpolated. Needs to be (numpy) universal function which shall be interpolated. If `arr` is an :class:`np.ndarray` with shape ``arr.shape == (N,spatial_dimension)``, the signature needs to be ``fct(arr) -> res``, where ``res`` is an :class:`np.ndarray` with shape ``(N,)``.
    :type fct: Callable

    :param spatial_dimension: dimension of the domain space.
    :type spatial_dimension: int
    :param poly_degree: degree of the interpolation polynomials
    :type poly_degree: int
    :param lp_degree: degree of the :math:`l_p` norm used to determine the `poly_degree`.
    :type lp_degree: int

    :return: The interpolant of ``fct`` using the default interpolator build from ``(spatial_dimension, poly_degree, lp_degree)``.
    :rtype: Interpolant
    """
    return Interpolant.from_degree(fct, spatial_dimension, poly_degree, lp_degree)
