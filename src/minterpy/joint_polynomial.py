#!/usr/bin/env python
""" classes for representing the sum of multiple polynomial instances at once
"""

from typing import Iterable, List

import numpy as np

__all__ = ['JointPolynomial']

__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"

from minterpy.multivariate_polynomial_abstract import MultivariatePolynomialSingleABC, MultivariatePolynomialABC
from minterpy.verification import rectify_query_points


def eval_each(polynomials: List[MultivariatePolynomialSingleABC], arg) -> Iterable[float]:
    for p in polynomials:
        yield p(arg)


def nr_mons_of_each(polynomials: List[MultivariatePolynomialSingleABC]) -> Iterable[int]:
    for p in polynomials:
        yield p.nr_of_monomials


def acc_nr_mons_of_each(polynomials: List[MultivariatePolynomialSingleABC]) -> Iterable[int]:
    # accumulated:
    acc = 0
    for nr_of_monomials in nr_mons_of_each(polynomials):
        acc += nr_of_monomials
        yield acc


class JointPolynomial(MultivariatePolynomialABC):
    """ class for combining multiple polynomials into a single polynomial instance

    NOTE: represents the SUM of all given sub-polynomials

    TODO: rename
    TODO: test
    """

    expected_type = MultivariatePolynomialSingleABC

    def __init__(self, sub_polynomials: Iterable[MultivariatePolynomialSingleABC]):
        # TODO check if the shape of the coefficients (all but the first axis) match!
        #  otherwise e.g. concatenation won't work!
        # TODO if all input polynomials are Lagrange polynomials.
        #  a LagrangeJointPolynomial must be constructed automatically!

        self.sub_polynomials: List[MultivariatePolynomialSingleABC] = []
        spatial_dimension = None
        for poly in sub_polynomials:
            if not issubclass(type(poly), self.expected_type):
                raise TypeError(
                    f'the given input polynomial {poly} is not a subclass of the expected {self.expected_type}')
            if spatial_dimension is None:
                spatial_dimension = poly.spatial_dimension
            elif poly.spatial_dimension != spatial_dimension:
                raise ValueError('all input polynomials must be of the same dimensionality'
                                 f' ({poly.spatial_dimension} != {spatial_dimension}')
            self.sub_polynomials.append(poly)
        if len(self.sub_polynomials) == 0:
            raise ValueError('A joint polynomial must consist of at least one polynomial, '
                             'but no polynomial has been given.')

    def _eval(self, arg) -> float:
        """
        :param arg: the point to evaluate the polynomial on
        :return: the value of the joint polynomial at `arg` = the sum of all sub polynomial values
        """
        return sum(eval_each(self.sub_polynomials, arg))

    @property
    def nr_of_monomials(self):
        """
        :return: the total amount of monomials of all sub polynomials
        """
        return sum(nr_mons_of_each(self.sub_polynomials))

    @property
    def spatial_dimension(self):
        """ the dimensionality of the polynomial
        """
        return self.sub_polynomials[0].spatial_dimension

    @property
    def coeffs(self) -> np.ndarray:
        """
        :return: the concatenated coefficients of all sub-polynomials
        """
        out = None
        for i, poly in enumerate(self.sub_polynomials):
            coeffs = poly.coeffs
            if coeffs is None:
                raise ValueError('trying to access uninitialized polynomial (coefficients are `None`)')
            if i == 0:
                out = coeffs
            else:
                out = np.append(out, coeffs, axis=0)
        return out

    @coeffs.setter
    def coeffs(self, value: np.ndarray):
        """ splits up and assigns the coefficients of each sub-polynomial

        NOTE: important for the regression which is setting the coefficients of the interpolating polynomial

        :param value:
            TODO perhaps also allow a list of coefficients for each polynomial
        """
        if value.shape[-1] != self.nr_of_monomials:
            raise ValueError(f'the given coefficients with shape {value.shape} do not fit '
                             f'the total amount of monomials {self.nr_of_monomials}')
        split_positions = np.fromiter(acc_nr_mons_of_each(self.sub_polynomials), dtype=int)
        # NOTE: do not pass the last split position (would result in an empty last split)
        split_coeffs = np.split(value, split_positions[:-1])
        for coeffs, poly in zip(split_coeffs, self.sub_polynomials):
            poly.coeffs = coeffs

    @property
    def unisolvent_nodes(self):
        """ the points the polynomial is defined on
        """
        return np.concatenate([p.grid.unisolvent_nodes for p in self.sub_polynomials], axis=0)

    @property
    def newt_coeffs_lagr_monomials(self):
        raise NotImplementedError
        # TODO ATTENTION: different shapes and sets of multi indices!

    def eval_lagrange_monomials_on(self, x):
        """ computes the values of all Lagrange monomials at all k input points

        NOTE: required for computing the regression transformation matrices
        NOTE: only supported by sub polynomials of type LagrangePolynomial!

        :param x: (m x k) the k points to evaluate on.
        :return: (k x N) the value of each Lagrange monomial in Newton form of all sub-polynomials at each point.
        """
        out = None
        nr_of_points, x = rectify_query_points(x, self.spatial_dimension)
        for i, poly in enumerate(self.sub_polynomials):
            # NOTE: only supported by LagrangePolynomials
            mon_vals = poly.eval_lagrange_monomials_on(x)
            mon_vals = mon_vals.reshape((nr_of_points, poly.nr_of_monomials))
            if i == 0:
                out = mon_vals
            else:
                out = np.append(out, mon_vals, axis=0)
        return out

# class JointLagrangePolynomial(JointPolynomial):
#     """ class for combining multiple Lagrange polynomials into a single polynomial instance
#     """
#     expected_type = LagrangePolynomial
#
#     def __init__(self, sub_polynomials: Iterable[LagrangePolynomial]):
#         super().__init__(sub_polynomials)
