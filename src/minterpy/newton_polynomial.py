"""
Module of the NewtonPolyomial class
"""

from minterpy.global_settings import DEBUG
from minterpy.multivariate_polynomial_abstract import MultivariatePolynomialSingleABC
from minterpy.utils import newt_eval
from minterpy.verification import verify_domain

__all__ = ["NewtonPolynomial"]


def dummy():
    # move dummy to util?
    raise NotImplementedError(f"This feature is not implemented yet.")


def newton_eval(newton_poly, x):
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
    """
    Newton polynomial class.
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
