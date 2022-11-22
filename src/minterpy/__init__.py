"""
This is the minterpy package init.

isort:skip_file
"""

from .version import version as __version__

__all__ = [
    "__version__",
]

from . import core  # noqa
from .core import *  # noqa

__all__ += core.__all__


from . import polynomials  # noqa
from .polynomials import *  # noqa

__all__ += polynomials.__all__


from . import transformations  # noqa
from .transformations import *  # noqa

__all__ += transformations.__all__


from . import interpolation  # noqa
from .interpolation import *  # noqa

__all__ += interpolation.__all__

from . import extras
from .extras import regression
from .extras.regression import *

__all__ += regression.__all__
