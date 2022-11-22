"""
Abstract base class for polynomial regression classes.

This module contains the abstract base class from which all concrete
implementations of polynomial regression classes should be created.
Such an abstract class ensures all the polynomial regression classes shares a set of common interface,
while their implementation may differs or additional methods or attributes may be defined.
"""

import abc


__all__ = ["RegressionABC"]


class RegressionABC(abc.ABC):
    """The abstract base class for all regression models."""

    @abc.abstractmethod
    def fit(self, xx, yy, *args, **kwargs):  # pragma: no cover
        """Abstract container for fitting a polynomial regression."""
        pass

    @abc.abstractmethod
    def predict(self, xx):  # pragma: no cover
        """Abstract container for making prediction using a polynomial regression."""
        pass

    @abc.abstractmethod
    def show(self):  # pragma: no cover
        """Abstract container for printing out the details of a polynomial regression model."""
        pass

    def __call__(self, xx):  # pragma: no cover
        """Evaluation of the polynomial regression model."""
        return self.predict(xx)
