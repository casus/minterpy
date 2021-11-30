"""
Module of the NewtonPolyomial class

.. todo::
    - implement staticmethods for Newton polynomials (or at least trasform them to another base).
"""

from minterpy.global_settings import DEBUG
from minterpy.utils import newt_eval

from ..core.ABC.multivariate_polynomial_abstract import \
    MultivariatePolynomialSingleABC
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


# TODO redundant
newton_generate_internal_domain = verify_domain
newton_generate_user_domain = verify_domain


class NewtonPolynomial(MultivariatePolynomialSingleABC):
    """Datatype to discribe polynomials in Newton base.

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

    generate_internal_domain = staticmethod(newton_generate_internal_domain)
    generate_user_domain = staticmethod(newton_generate_user_domain)
