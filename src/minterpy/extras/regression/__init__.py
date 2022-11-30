"""
The submodule `regression` is part of `minterpy-extras`.

It contains functions and classes for multivariate polynomial regression.
"""

__all__ = []

from . import regression_abc  # noqa
from .regression_abc import *

from . import ordinary_regression
from .ordinary_regression import *

__all__ += regression_abc.__all__
__all__ += ordinary_regression.__all__

