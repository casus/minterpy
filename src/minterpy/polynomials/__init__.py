"""
The submodule `polynomials` is part of `minterpy`.

It contains concrete implementations for multivariate canonical, Newton and Lagrange polynomial base, respectively.
"""

__all__ = []

from . import canonical_polynomial  # noqa
from .canonical_polynomial import *  # noqa

__all__ += canonical_polynomial.__all__

from . import newton_polynomial  # noqa
from .newton_polynomial import *  # noqa

__all__ += newton_polynomial.__all__

from . import lagrange_polynomial  # noqa
from .lagrange_polynomial import *  # noqa

__all__ += lagrange_polynomial.__all__
