"""
This is the core module. It should not be touched unless you know exactly what are you doing.
"""

__all__ = []

from . import multi_index  # noqa
from .multi_index import *  # noqa

__all__ += multi_index.__all__

from . import grid  # noqa
from .grid import *  # noqa

__all__ += grid.__all__

from . import ABC  # noqa # ABCs are not exposed to the top level!
from . import tree  # noqa
from . import utils  # noqa # core utils are not exposed to top level
