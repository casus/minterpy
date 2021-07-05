"""
This is the minterpy package init.

isort:skip_file
"""

from .version import version as __version__

__all__ = ["__version__",]

from . import core
from .core import *

__all__+=core.__all__


from . import polynomials
from .polynomials import *

__all__+=polynomials.__all__


from . import transformations
from .transformations import *

__all__+=transformations.__all__


from . import interpolation
from .interpolation import *

__all__+=interpolation.__all__



"""
from minterpy.multivariate_polynomial_abstract import *
from minterpy.multi_index import *
from minterpy.canonical_polynomial import *
from minterpy.newton_polynomial import *
from minterpy.multi_index_tree import *
from minterpy.grid import *
from minterpy.transformation_abstract import *
from minterpy.transformation_newton import *
from minterpy.lagrange_polynomial import *
from minterpy.transformation_canonical import *

"""
