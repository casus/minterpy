"""
Module of the NewtonPolyomial class

.. todo::
    - implement staticmethods for Newton polynomials (or at least trasform them to another base).
"""
import numpy as np

from minterpy.global_settings import DEBUG
from minterpy.utils import newt_eval, deriv_newt_eval
from minterpy.dds import dds

from ..core.ABC.multivariate_polynomial_abstract import MultivariatePolynomialSingleABC
from ..core.verification import verify_domain

__all__ = ["NewtonPolynomial"]


def dummy():
    """Placeholder function.

    .. warning::
      This feature is not implemented yet!
    """
    # move dummy to util?
    raise NotImplementedError("This feature is not implemented yet.")


def newton_eval(newton_poly, x):
    """Evaluation function in Newton base.

    This is a wrapper for the evaluation function in Newton base.

    :param newton_poly: The :class:`NewtonPolynomial` which is evaluated.
    :type newton_poly: NewtonPolynomial
    :param x: The argument(s) the Newton polynomial shall be evaluated. The input shape needs to be ``(N,dim)``, where ``N`` refered to the number of points and ``dim`` refers to the dimension of the domain space, i.e. the coordinates of the argument vector.


    .. todo::
        - check right order of axes for ``x``.

    See Also
    --------
    newt_eval : comcrete implementation of the evaluation in Newton base.

    """
    return newt_eval(
        x,
        newton_poly.coeffs,
        newton_poly.multi_index.exponents,
        newton_poly.grid.generating_points,
        verify_input=DEBUG,
    )


def _newton_partial_diff(poly: "NewtonPolynomial", dim: int, order: int) -> "NewtonPolynomial":
    """ Partial differentiation in Newton basis.

    Notes
    -----
    Performs a transformation from Lagrange to Newton using DDS.
    """
    spatial_dim = poly.multi_index.spatial_dimension
    deriv_order_along = [0]*spatial_dim
    deriv_order_along[dim] = order
    return _newton_diff(poly, deriv_order_along)

def _newton_diff(poly: "NewtonPolynomial", order: np.ndarray) -> "NewtonPolynomial":
    """ Partial differentiation in Newton basis.

    Notes
    -----
    Performs a transformation from Lagrange to Newton using DDS.
    """

    # When you evaluate the derivatives on the unisolvent nodes, you get the coefficients for
    # the derivative polynomial in Lagrange basis.
    lag_coeffs = deriv_newt_eval(poly.grid.unisolvent_nodes, poly.coeffs,
                                 poly.grid.multi_index.exponents,
                                 poly.grid.generating_points, order)

    # DDS returns a 2D array, converting it to 1d
    newt_coeffs = dds(lag_coeffs, poly.grid.tree).reshape(-1)

    return NewtonPolynomial(coeffs=newt_coeffs, multi_index=poly.multi_index,
                              grid=poly.grid)

# TODO redundant
newton_generate_internal_domain = verify_domain
newton_generate_user_domain = verify_domain


class NewtonPolynomial(MultivariatePolynomialSingleABC):
    """Datatype to describe polynomials in Newton base.

    For a definition of the Newton base, see the mathematical introduction.

    .. todo::
        - provide a short definition of this base here.

    Attributes
    ----------
    coeffs
    nr_active_monomials
    spatial_dimension
    unisolvent_nodes


    """

    # Virtual Functions
    _add = staticmethod(dummy)
    _sub = staticmethod(dummy)
    _mul = staticmethod(dummy)
    _div = staticmethod(dummy)
    _pow = staticmethod(dummy)
    _eval = newton_eval

    _partial_diff = staticmethod(_newton_partial_diff)
    _diff = staticmethod(_newton_diff)

    generate_internal_domain = staticmethod(newton_generate_internal_domain)
    generate_user_domain = staticmethod(newton_generate_user_domain)
